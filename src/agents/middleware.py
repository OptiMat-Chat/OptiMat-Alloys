"""Agent Framework middleware for OptiMat model clients.

Carries the OpenRouter rate-limit pacing over from the AutoGen-era
RateLimitedOpenAIChatCompletionClient subclass. Chat middleware runs per
model call (every iteration of the tool-calling loop), so the pacing
guarantees are identical to the old per-create()/create_stream() delay.
"""

from agent_framework import chat_middleware

import src.core.openrouter_client as openrouter_client


@chat_middleware
async def openrouter_rate_limit(context, call_next):
    """Wait out the OpenRouter pacing window (3 s, 20 req/min) before the call.

    AF middleware contract: call_next() returns None; the model response
    flows through context.result. Attribute access at call time keeps the
    shared rate-limiter state in src.core.openrouter_client patchable.
    """
    await openrouter_client.wait_before_openrouter_request()
    await call_next()
