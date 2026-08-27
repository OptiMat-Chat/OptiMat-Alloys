"""
Scientist agent implementation for atomistic simulations.

This module provides the Scientist agent that assists users with
materials science simulations and analysis.
"""

from typing import Any, List, Callable, Optional

from agent_framework import Agent

from .base import BaseAgent, AgentConfig

# Cap on model roundtrips within one run's tool-calling loop. Parity with
# the AutoGen-era RoundRobinGroupChat(max_turns=10): one round-robin turn
# was one model call, so a single user message was bounded by 10 calls.
# On exhaustion the framework forces a final text answer (tool_choice="none").
MAX_TOOL_ITERATIONS = 10


class ScientistAgent(BaseAgent):
    """
    Scientist agent specialized in atomistic simulations.

    Provides access to tools for generating and analyzing atomic structures,
    computing properties, and visualizing results.
    """

    def create_agent(self) -> Agent:
        """
        Create an Agent Framework Agent configured as a Scientist.

        The agent is built WITHOUT constructor instructions: run_chat
        delivers config.system_message per run (optionally prefixed with
        session context), which keeps the run-level value the sole system
        message. Conversation history lives in the AgentSession.

        Returns:
            Configured agent_framework.Agent instance

        Examples:
            >>> config = AgentConfig(
            ...     name="Scientist",
            ...     system_message="You are a scientist",
            ...     model_client=client
            ... )
            >>> agent = ScientistAgent(config)  # doctest: +SKIP
            >>> instance = agent.create_agent()  # doctest: +SKIP
        """
        client = self.config.model_client
        # Official AF idiom for configuring the tool loop on an existing
        # client (see agent_framework._tools docs): mutate the config dict.
        client.function_invocation_configuration["max_iterations"] = (
            MAX_TOOL_ITERATIONS
        )
        return Agent(
            client=client,
            name=self.config.name,
            tools=self.config.tools,
            default_options={"temperature": self.config.temperature},
        )


def create_scientist_agent(
    model_client: Any,
    tools: List[Callable],
    name: str = "Scientist",
    system_message: Optional[str] = None,
    temperature: float = 0.0,
) -> ScientistAgent:
    """
    Convenience function to create a Scientist agent.

    Args:
        model_client: Agent Framework chat client for the agent
        tools: List of tool functions
        name: Agent name
        system_message: Custom system message (uses default if None)
        temperature: Sampling temperature

    Returns:
        Configured ScientistAgent

    Examples:
        >>> agent = create_scientist_agent(  # doctest: +SKIP
        ...     model_client=client,
        ...     tools=[generate_alloy_supercell]
        ... )
    """
    # Import the canonical system message from the factory to maintain a single source of truth
    from .factory import AgentFactory
    default_message = AgentFactory.DEFAULT_MESSAGES["scientist"]

    config = AgentConfig(
        name=name,
        system_message=system_message or default_message,
        tools=tools,
        model_client=model_client,
        temperature=temperature,
    )

    return ScientistAgent(config)
