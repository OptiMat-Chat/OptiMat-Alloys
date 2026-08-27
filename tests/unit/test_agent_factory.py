"""Agent layer parity tests.

The Scientist agent moves from AutoGen AssistantAgent to agent_framework
Agent with three frozen contracts:
1. System-message texts are byte-identical (SHA-256 pinned).
2. The Agent is built WITHOUT constructor instructions — run_chat delivers
   the (possibly context-prefixed) system message per run, so the run-level
   value is the sole system message and history lives in the AgentSession.
3. temperature rides in default_options; the tool loop is capped at 10
   model roundtrips (parity with the old RoundRobinGroupChat max_turns=10).
"""

import hashlib

from agent_framework import Agent

from src.agents.factory import AgentFactory
from src.agents.base import AgentConfig
from src.agents.local_models.ollama_factory import create_ollama_client


def make_client():
    return create_ollama_client("qwen3:4b")


def make_scientist(**overrides):
    kwargs = dict(
        model_client=make_client(),
        tools=[_demo_tool],
        name="Scientist",
    )
    kwargs.update(overrides)
    return AgentFactory.create_scientist(**kwargs)


async def _demo_tool(x: str) -> str:
    """Demo tool."""
    return x


class TestFrozenSystemMessages:
    SCIENTIST_SHA256 = (
        "b00b5d1cc5e48d1ea2935291537a4982c64227bf78e59be800d485e0e75cc4bc"
    )
    ASSISTANT_SHA256 = (
        "114ddbd17e9ad69e6ab4e9a5da80dce1f690d8de71b5960ea235c5d1304cd5b5"
    )

    def test_scientist_message_byte_identical(self):
        digest = hashlib.sha256(
            AgentFactory.DEFAULT_MESSAGES["scientist"].encode("utf-8")
        ).hexdigest()
        assert digest == self.SCIENTIST_SHA256, (
            "Scientist system message changed. The migration-era freeze is lifted "
            "(stage 2), but this hash still guards against UNINTENDED drift — "
            "update it only alongside a deliberate, reviewed prompt change."
        )

    def test_assistant_message_byte_identical(self):
        digest = hashlib.sha256(
            AgentFactory.DEFAULT_MESSAGES["assistant"].encode("utf-8")
        ).hexdigest()
        assert digest == self.ASSISTANT_SHA256

    def test_get_default_scientist_message_is_the_frozen_text(self):
        assert (
            AgentFactory.get_default_scientist_message()
            == AgentFactory.DEFAULT_MESSAGES["scientist"]
        )

    def test_dynamic_message_format_unchanged(self):
        combined = AgentFactory.build_dynamic_system_message("BASE", "CONTEXT")
        assert combined == "CONTEXT\n\n---\n\nBASE"

    def test_dynamic_message_empty_context_passthrough(self):
        assert AgentFactory.build_dynamic_system_message("BASE", "") == "BASE"
        assert AgentFactory.build_dynamic_system_message("BASE", "   ") == "BASE"


class TestScientistAgentConstruction:
    def test_returns_af_agent_named_scientist(self):
        agent = make_scientist().get_agent()
        assert isinstance(agent, Agent)
        assert agent.name == "Scientist"

    def test_no_constructor_instructions(self):
        # Instructions are delivered per run (sole system message); baking
        # them into the Agent would cause run-level instructions to APPEND
        # instead, breaking text parity. Agent folds constructor args into
        # default_options (ChatOptions), so absence is asserted there.
        agent = make_scientist().get_agent()
        assert agent.default_options.get("instructions") is None

    def test_system_message_kept_in_config(self):
        wrapper = make_scientist()
        assert (
            wrapper.config.system_message
            == AgentFactory.DEFAULT_MESSAGES["scientist"]
        )

    def test_tools_attached(self):
        agent = make_scientist().get_agent()
        assert len(agent.default_options["tools"]) == 1

    def test_temperature_in_default_options(self):
        agent = make_scientist().get_agent()
        assert agent.default_options["temperature"] == 0.0

    def test_custom_temperature_flows_through(self):
        agent = make_scientist(temperature=0.7).get_agent()
        assert agent.default_options["temperature"] == 0.7

    def test_tool_loop_capped_at_10_iterations(self):
        wrapper = make_scientist()
        wrapper.get_agent()
        fic = wrapper.config.model_client.function_invocation_configuration
        assert fic["max_iterations"] == 10

    def test_get_agent_caches_instance(self):
        wrapper = make_scientist()
        assert wrapper.get_agent() is wrapper.get_agent()


class TestConfigUpdates:
    def test_update_system_message_mutates_config(self):
        wrapper = make_scientist()
        wrapper.get_agent()
        wrapper.update_system_message("NEW MESSAGE")
        assert wrapper.config.system_message == "NEW MESSAGE"

    def test_update_tools_rebuilds_agent(self):
        wrapper = make_scientist()
        first = wrapper.get_agent()

        async def other_tool(y: int) -> int:
            """Other tool."""
            return y

        wrapper.update_tools([_demo_tool, other_tool])
        rebuilt = wrapper.get_agent()
        assert rebuilt is not first
        assert len(rebuilt.default_options["tools"]) == 2

    def test_agent_config_has_no_autogen_relics(self):
        fields = set(AgentConfig.__dataclass_fields__)
        assert "model_context" not in fields
        assert "reflect_on_tool_use" not in fields
        assert "model_client_stream" not in fields
