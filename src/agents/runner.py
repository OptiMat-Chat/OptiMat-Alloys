"""Chainlit-free conversation runner for the Scientist agent.

Reproduces the AutoGen-era team semantics over agent_framework:

- RoundRobinGroupChat([scientist], max_turns=10) re-invoked the single
  agent until TextMentionTermination("?") saw a question mark in a
  Scientist message. Here: one agent.run() per round, re-invoked with no
  new user message while the round's final text lacks "?", hard-capped at
  MAX_ROUNDS.
- ModelClientStreamingChunkEvent tokens streamed into the UI, and any
  non-chunk event finalized the open UI message. Here: text deltas go to
  on_token(text, author); updates carrying TOOL activity (function_call /
  function_result content) trigger on_segment_end(), as does the end of
  each run. Other non-text content (reasoning deltas, usage frames) is
  invisible — the AutoGen UI never saw those as events either, so they
  must not split a streaming message.

Intentional refinement vs AutoGen: termination evaluates only the LAST
assistant text message of a round (the user-visible answer) — never raw
tool payloads that happen to contain "?" (an AutoGen quirk), and never
pre-tool commentary mid-chain.

Budget note: the old team capped a user message at 10 model calls total
(tool steps consumed turns). Now each round's tool loop is itself capped
at MAX_TOOL_ITERATIONS=10 (+1 forced final), so the theoretical worst
case is MAX_ROUNDS×11 calls — reachable only if the model runs ten full
tool cascades AND never asks a question. Realistic no-question rounds are
single text-only calls, exactly like the old re-invocation. Tune
MAX_ROUNDS/MAX_TOOL_ITERATIONS if that ceiling ever matters in practice.

UI callbacks are injected, keeping this module fully unit-testable; the
Chainlit glue lives in run_chat.py.
"""

from typing import Awaitable, Callable, Optional

from .base import BaseAgent
from .scientist import MAX_TOOL_ITERATIONS

# Cap on agent re-invocations for a single user message — parity with the
# AutoGen-era RoundRobinGroupChat(max_turns=10).
MAX_ROUNDS = 10

OnToken = Callable[[str, str], Awaitable[None]]
OnSegmentEnd = Callable[[], Awaitable[None]]

__all__ = ["MAX_ROUNDS", "MAX_TOOL_ITERATIONS", "run_until_question"]


async def run_until_question(
    wrapper: BaseAgent,
    session,
    user_text: str,
    *,
    context_block: Optional[str] = None,
    on_token: OnToken,
    on_segment_end: OnSegmentEnd,
) -> str:
    """Run the agent on a user message until it asks the user a question.

    The system message is composed HERE from wrapper.config.system_message
    (optionally prefixed with the session context block, same format as the
    AutoGen era) and delivered per run — the agent holds no constructor
    instructions, so this is the sole system message while conversation
    history accumulates in the session.

    Args:
        wrapper: agent wrapper (ScientistAgent); its config provides the
            base system message and the built agent.
        session: AgentSession carrying conversation history.
        user_text: the new user message for round 1; continuation rounds
            re-invoke the agent without a new message.
        context_block: optional SessionState context block to prefix.
        on_token: awaited per streamed text delta with (text, author).
        on_segment_end: awaited whenever the current UI message should be
            finalized (tool-call boundary or end of a run).

    Returns:
        The final assistant text of the last round.
    """
    from .factory import AgentFactory

    base_message = wrapper.config.system_message
    instructions = (
        AgentFactory.build_dynamic_system_message(base_message, context_block)
        if context_block
        else base_message
    )
    agent = wrapper.get_agent()

    final_text = ""
    messages = user_text
    for _round in range(MAX_ROUNDS):
        response = await _stream_one_run(
            agent, session, messages, instructions, on_token, on_segment_end
        )
        final_text = _last_assistant_text(response)
        if "?" in final_text:
            break
        # Continuation: no new user message — the RoundRobin re-invocation
        # equivalent. The model keeps going from session history.
        messages = None
    return final_text


def _last_assistant_text(response) -> str:
    """The user-visible answer of a run: its last assistant text message.

    Pre-tool commentary and tool payloads must not drive termination, so
    neither response.text (which concatenates ALL assistant text in the
    run) nor tool-result content qualifies.
    """
    for message in reversed(response.messages or []):
        role = getattr(message.role, "value", message.role)
        if role == "assistant" and message.text:
            return message.text
    return ""


async def _stream_one_run(
    agent,
    session,
    messages,
    instructions: str,
    on_token: OnToken,
    on_segment_end: OnSegmentEnd,
):
    """One agent.run(): stream deltas to the callbacks, return the response."""
    stream = agent.run(
        messages,
        session=session,
        stream=True,
        options={"instructions": instructions},
    )
    try:
        async for update in stream:
            text = update.text
            if text:
                await on_token(text, update.author_name or agent.name)
            if any(
                content.type in ("function_call", "function_result")
                for content in (update.contents or [])
            ):
                await on_segment_end()
    except BaseException:
        # Flush the open UI message on the error path too, but never let a
        # failing flush mask the original (classifiable) provider error.
        try:
            await on_segment_end()
        except Exception:
            pass
        raise
    await on_segment_end()
    return await stream.get_final_response()
