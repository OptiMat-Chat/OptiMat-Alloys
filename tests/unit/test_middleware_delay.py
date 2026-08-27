"""The OpenRouter rate-limit delay must run before every model call.

Replaces the AutoGen-era RateLimitedOpenAIChatCompletionClient subclass:
the same `wait_before_openrouter_request()` (3 s pacing, 20 req/min cap)
now runs as Agent Framework chat middleware, which fires per model call —
including every iteration of the tool-calling loop, matching the old
"delay before create() AND create_stream()" behavior.
"""

import pytest

import src.core.openrouter_client as openrouter_client
from src.agents.middleware import openrouter_rate_limit


class _Recorder:
    def __init__(self):
        self.calls = []

    async def wait(self):
        self.calls.append("wait")

    async def call_next(self):
        self.calls.append("call_next")
        return "downstream-result"


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(
        openrouter_client, "wait_before_openrouter_request", rec.wait
    )
    return rec


async def test_delay_awaited_before_model_call(recorder):
    await openrouter_rate_limit(context=object(), call_next=recorder.call_next)
    assert recorder.calls == ["wait", "call_next"]


async def test_delay_runs_again_on_every_call(recorder):
    for _ in range(3):
        await openrouter_rate_limit(context=object(), call_next=recorder.call_next)
    assert recorder.calls == ["wait", "call_next"] * 3


async def test_downstream_exception_propagates(recorder):
    async def failing_call_next():
        recorder.calls.append("call_next")
        raise RuntimeError("provider exploded")

    with pytest.raises(RuntimeError, match="provider exploded"):
        await openrouter_rate_limit(context=object(), call_next=failing_call_next)
    # The delay still ran first; the error was not swallowed.
    assert recorder.calls == ["wait", "call_next"]


async def test_delay_failure_prevents_model_call(recorder, monkeypatch):
    async def failing_wait():
        recorder.calls.append("wait")
        raise RuntimeError("rate limiter broke")

    monkeypatch.setattr(
        openrouter_client, "wait_before_openrouter_request", failing_wait
    )
    with pytest.raises(RuntimeError, match="rate limiter broke"):
        await openrouter_rate_limit(context=object(), call_next=recorder.call_next)
    assert recorder.calls == ["wait"]
