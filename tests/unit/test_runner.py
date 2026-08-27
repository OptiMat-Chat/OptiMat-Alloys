"""Runner parity tests: streaming adapter, the "?" termination loop, and
session/instructions plumbing — the behaviors RoundRobinGroupChat +
TextMentionTermination("?") + ModelClientStreamingChunkEvent provided
under AutoGen, now reproduced over agent_framework with no Chainlit
dependency (callbacks injected).
"""

import pytest

from agent_framework import Agent

from src.agents.factory import AgentFactory
from src.agents.runner import MAX_ROUNDS, run_until_question

from tests.unit.fake_chat_client import FakeChatClient, text_update, tool_call_update


class Sink:
    """Collects streaming callbacks the way run_chat's Chainlit glue will."""

    def __init__(self):
        self.events = []

    async def on_token(self, text, author):
        self.events.append(("token", text, author))

    async def on_segment_end(self):
        self.events.append(("segment_end",))

    @property
    def tokens(self):
        return [e[1] for e in self.events if e[0] == "token"]

    @property
    def streamed_text(self):
        return "".join(self.tokens)


TOOL_RUNS = []


async def fake_tool(x: str) -> str:
    """Record invocation; echo input."""
    TOOL_RUNS.append(x)
    return f"tool-saw-{x}"


async def questioning_tool(x: str) -> str:
    """A tool whose raw output contains question marks."""
    return "??? does this terminate ???"


def make_wrapper(scripts, tools=(fake_tool,)):
    client = FakeChatClient(scripts)
    wrapper = AgentFactory.create_scientist(
        model_client=client, tools=list(tools), name="Scientist"
    )
    return wrapper, client


def session_for(wrapper):
    return wrapper.get_agent().create_session()


@pytest.fixture(autouse=True)
def _clear_tool_runs():
    TOOL_RUNS.clear()


class TestStreamingAdapter:
    async def test_tokens_stream_in_order_with_author(self):
        wrapper, client = make_wrapper(
            [[text_update("Hel"), text_update("lo"), text_update(" world?")]]
        )
        sink = Sink()
        await run_until_question(
            wrapper,
            session_for(wrapper),
            "hi",
            on_token=sink.on_token,
            on_segment_end=sink.on_segment_end,
        )
        assert sink.streamed_text == "Hello world?"
        assert all(e[2] == "Scientist" for e in sink.events if e[0] == "token")

    async def test_reasoning_chunks_do_not_split_the_message(self):
        # OpenRouter reasoning models interleave text_reasoning content with
        # text deltas. The AutoGen-era UI never saw reasoning chunks, so one
        # answer stayed one message — only TOOL activity closes a segment.
        from agent_framework import ChatResponseUpdate, Content

        reasoning_update = ChatResponseUpdate(
            contents=[Content.from_text_reasoning(text="thinking hard")],
            role="assistant",
            author_name="Scientist",
        )
        wrapper, client = make_wrapper(
            [[text_update("Part one. "), reasoning_update, text_update("Part two?")]]
        )
        sink = Sink()
        await run_until_question(
            wrapper,
            session_for(wrapper),
            "hi",
            on_token=sink.on_token,
            on_segment_end=sink.on_segment_end,
        )
        # No segment boundary between the text deltas — a consumer holding
        # an open message would render exactly one message.
        boundary_positions = [
            i for i, e in enumerate(sink.events) if e == ("segment_end",)
        ]
        token_positions = [i for i, e in enumerate(sink.events) if e[0] == "token"]
        assert boundary_positions == [len(sink.events) - 1]
        assert max(token_positions) < boundary_positions[0]
        # Reasoning text never leaks into the visible stream.
        assert "thinking hard" not in sink.streamed_text

    async def test_run_always_flushes_at_end(self):
        wrapper, client = make_wrapper([[text_update("Done?")]])
        sink = Sink()
        await run_until_question(
            wrapper,
            session_for(wrapper),
            "hi",
            on_token=sink.on_token,
            on_segment_end=sink.on_segment_end,
        )
        assert sink.events[-1] == ("segment_end",)

    async def test_tool_call_closes_streaming_segment_and_executes_tool(self):
        wrapper, client = make_wrapper(
            [
                # model call 1: text, then a tool call
                [
                    text_update("Let me check."),
                    tool_call_update("c1", "fake_tool", {"x": "iron"}),
                ],
                # model call 2 (after tool result): final answer
                [text_update("Iron looks good — continue?")],
            ]
        )
        sink = Sink()
        final = await run_until_question(
            wrapper,
            session_for(wrapper),
            "check iron",
            on_token=sink.on_token,
            on_segment_end=sink.on_segment_end,
        )
        # The real FunctionInvocationLayer executed our tool
        assert TOOL_RUNS == ["iron"]
        assert len(client.calls) == 2
        # Pre-tool text streamed, then the segment closed at the call boundary
        first_boundary = sink.events.index(("segment_end",))
        streamed_before = "".join(
            e[1] for e in sink.events[:first_boundary] if e[0] == "token"
        )
        assert streamed_before == "Let me check."
        assert "continue?" in sink.streamed_text
        assert "continue?" in final


class TestQuestionLoop:
    async def test_stops_on_first_question(self):
        wrapper, client = make_wrapper([[text_update("Want more?")]])
        sink = Sink()
        final = await run_until_question(
            wrapper,
            session_for(wrapper),
            "hi",
            on_token=sink.on_token,
            on_segment_end=sink.on_segment_end,
        )
        assert final == "Want more?"
        assert len(client.calls) == 1

    async def test_reinvokes_until_question_without_new_user_message(self):
        wrapper, client = make_wrapper(
            [
                [text_update("Statement one.")],
                [text_update("Statement two.")],
                [text_update("Finally a question?")],
            ]
        )
        sink = Sink()
        final = await run_until_question(
            wrapper,
            session_for(wrapper),
            "go",
            on_token=sink.on_token,
            on_segment_end=sink.on_segment_end,
        )
        assert final == "Finally a question?"
        assert len(client.calls) == 3
        # Continuation rounds add no new user message (RoundRobin parity):
        # every model call sees exactly the one original user message —
        # and still receives the instructions on every round.
        for i in range(3):
            assert client.user_messages(i) == ["go"]
            assert (
                client.instructions(i)
                == AgentFactory.DEFAULT_MESSAGES["scientist"]
            )

    async def test_pre_tool_question_does_not_terminate(self):
        # "?" in commentary preceding a tool call is not the user-visible
        # answer; the round's LAST assistant text decides termination.
        wrapper, client = make_wrapper(
            [
                [
                    text_update("Should I check the DB? Searching now."),
                    tool_call_update("c1", "fake_tool", {"x": "iron"}),
                ],
                [text_update("Found it. No follow-up offered.")],
                [text_update("Anything else?")],
            ]
        )
        sink = Sink()
        final = await run_until_question(
            wrapper,
            session_for(wrapper),
            "check iron",
            on_token=sink.on_token,
            on_segment_end=sink.on_segment_end,
        )
        assert final == "Anything else?"
        assert len(client.calls) == 3

    async def test_question_inside_tool_result_does_not_terminate(self):
        # Old AutoGen quirk: TextMentionTermination could fire on raw tool
        # payloads. Tool output (here containing "?") must neither stream
        # to the UI nor terminate the loop.
        wrapper, client = make_wrapper(
            [
                [tool_call_update("c1", "questioning_tool", {"x": "hm"})],
                [text_update("Statement without question mark.")],
                [text_update("Now a real question?")],
            ],
            tools=(questioning_tool,),
        )
        sink = Sink()
        final = await run_until_question(
            wrapper,
            session_for(wrapper),
            "go",
            on_token=sink.on_token,
            on_segment_end=sink.on_segment_end,
        )
        assert final == "Now a real question?"
        assert len(client.calls) == 3
        assert "???" not in sink.streamed_text

    async def test_tool_output_never_leaks_into_stream(self):
        wrapper, client = make_wrapper(
            [
                [
                    text_update("Let me check."),
                    tool_call_update("c1", "fake_tool", {"x": "iron"}),
                ],
                [text_update("All good — continue?")],
            ]
        )
        sink = Sink()
        await run_until_question(
            wrapper,
            session_for(wrapper),
            "check iron",
            on_token=sink.on_token,
            on_segment_end=sink.on_segment_end,
        )
        assert "tool-saw-iron" not in sink.streamed_text

    async def test_returns_last_message_only_not_concatenation(self):
        wrapper, client = make_wrapper(
            [
                [
                    text_update("Pre-tool note."),
                    tool_call_update("c1", "fake_tool", {"x": "cu"}),
                ],
                [text_update("Copper is stable — more?")],
            ]
        )
        sink = Sink()
        final = await run_until_question(
            wrapper,
            session_for(wrapper),
            "copper",
            on_token=sink.on_token,
            on_segment_end=sink.on_segment_end,
        )
        assert final == "Copper is stable — more?"

    async def test_hard_cap_at_max_rounds(self):
        wrapper, client = make_wrapper(
            [[text_update(f"Statement {i}.")] for i in range(MAX_ROUNDS + 5)]
        )
        sink = Sink()
        await run_until_question(
            wrapper,
            session_for(wrapper),
            "go",
            on_token=sink.on_token,
            on_segment_end=sink.on_segment_end,
        )
        assert len(client.calls) == MAX_ROUNDS

    async def test_max_rounds_matches_old_team_cap(self):
        assert MAX_ROUNDS == 10


class TestInstructionsAndSession:
    async def test_base_system_message_is_sole_instructions(self):
        wrapper, client = make_wrapper([[text_update("Hi?")]])
        await run_until_question(
            wrapper,
            session_for(wrapper),
            "hi",
            on_token=_noop_token,
            on_segment_end=_noop_end,
        )
        assert client.instructions(0) == AgentFactory.DEFAULT_MESSAGES["scientist"]

    async def test_context_block_prefixes_exactly_like_autogen_era(self):
        wrapper, client = make_wrapper([[text_update("Hi?")]])
        await run_until_question(
            wrapper,
            session_for(wrapper),
            "hi",
            context_block="## SESSION CONTEXT\n- Calculator: orb-v3",
            on_token=_noop_token,
            on_segment_end=_noop_end,
        )
        expected = AgentFactory.build_dynamic_system_message(
            AgentFactory.DEFAULT_MESSAGES["scientist"],
            "## SESSION CONTEXT\n- Calculator: orb-v3",
        )
        assert client.instructions(0) == expected
        assert client.instructions(0).startswith("## SESSION CONTEXT")

    async def test_temperature_survives_run_options(self):
        wrapper, client = make_wrapper([[text_update("Hi?")]])
        await run_until_question(
            wrapper,
            session_for(wrapper),
            "hi",
            on_token=_noop_token,
            on_segment_end=_noop_end,
        )
        assert client.calls[0]["options"]["temperature"] == 0.0

    async def test_history_carries_across_turns_in_one_session(self):
        wrapper, client = make_wrapper(
            [[text_update("First answer?")], [text_update("Second answer?")]]
        )
        session = session_for(wrapper)
        await run_until_question(
            wrapper, session, "turn one",
            on_token=_noop_token, on_segment_end=_noop_end,
        )
        await run_until_question(
            wrapper, session, "turn two",
            on_token=_noop_token, on_segment_end=_noop_end,
        )
        # Second model call sees both user turns AND the first reply.
        assert client.user_messages(1) == ["turn one", "turn two"]
        all_texts = [m.text for m in client.calls[1]["messages"]]
        assert "First answer?" in all_texts

    async def test_fresh_session_resets_history(self):
        wrapper, client = make_wrapper(
            [[text_update("First answer?")], [text_update("Second answer?")]]
        )
        await run_until_question(
            wrapper, session_for(wrapper), "turn one",
            on_token=_noop_token, on_segment_end=_noop_end,
        )
        await run_until_question(
            wrapper, session_for(wrapper), "turn two",
            on_token=_noop_token, on_segment_end=_noop_end,
        )
        # Model-switch behavior: a new session means the model only sees
        # the new turn (history intentionally cleared, as the UI announces).
        assert client.user_messages(1) == ["turn two"]

    async def test_session_portable_across_agent_instances(self):
        # Settings updates rebuild the wrapper/agent; the session object is
        # what carries the conversation.
        wrapper_a, client = make_wrapper(
            [[text_update("A answered?")], [text_update("B answered?")]]
        )
        session = session_for(wrapper_a)
        await run_until_question(
            wrapper_a, session, "first",
            on_token=_noop_token, on_segment_end=_noop_end,
        )
        wrapper_b = AgentFactory.create_scientist(
            model_client=client, tools=[fake_tool], name="Scientist"
        )
        await run_until_question(
            wrapper_b, session, "second",
            on_token=_noop_token, on_segment_end=_noop_end,
        )
        assert client.user_messages(1) == ["first", "second"]


class TestErrorPaths:
    async def test_provider_error_propagates_after_flush(self):
        class ExplodingClient(FakeChatClient):
            def _inner_get_response(self, **kwargs):
                raise RuntimeError("429 provider rate limit")

        client = ExplodingClient()
        wrapper = AgentFactory.create_scientist(
            model_client=client, tools=[fake_tool], name="Scientist"
        )
        sink = Sink()
        with pytest.raises(Exception, match="429 provider rate limit"):
            await run_until_question(
                wrapper,
                session_for(wrapper),
                "hi",
                on_token=sink.on_token,
                on_segment_end=sink.on_segment_end,
            )
        # The open segment was still flushed on the error path.
        assert ("segment_end",) in sink.events

    async def test_failing_flush_does_not_mask_provider_error(self):
        # run_chat classifies errors by string; a broken websocket send in
        # the error-path flush must not replace the original exception.
        class ExplodingClient(FakeChatClient):
            def _inner_get_response(self, **kwargs):
                raise RuntimeError("429 provider rate limit")

        async def broken_segment_end():
            raise OSError("websocket already closed")

        client = ExplodingClient()
        wrapper = AgentFactory.create_scientist(
            model_client=client, tools=[fake_tool], name="Scientist"
        )
        with pytest.raises(Exception, match="429 provider rate limit"):
            await run_until_question(
                wrapper,
                session_for(wrapper),
                "hi",
                on_token=_noop_token,
                on_segment_end=broken_segment_end,
            )


async def _noop_token(text, author):
    pass


async def _noop_end():
    pass
