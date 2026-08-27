"""Scripted chat client for exercising the real Agent pipeline offline.

Subclasses the same layer stack as the production OpenAI client minus
telemetry (FunctionInvocationLayer → ChatMiddlewareLayer → BaseChatClient),
so agent.run() goes through genuine options merging, session history,
middleware, and tool execution — only the wire call is faked.
"""

import json
from typing import Any, Mapping, Sequence

from agent_framework import (
    BaseChatClient,
    ChatMiddlewareLayer,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionInvocationLayer,
    Message,
    ResponseStream,
)


def text_update(text: str, author: str = "Scientist") -> ChatResponseUpdate:
    """A streamed assistant text delta."""
    return ChatResponseUpdate(
        contents=[Content.from_text(text)],
        role="assistant",
        author_name=author,
    )


def tool_call_update(
    call_id: str, name: str, arguments: dict, author: str = "Scientist"
) -> ChatResponseUpdate:
    """A streamed function-call request from the model."""
    return ChatResponseUpdate(
        contents=[
            Content.from_function_call(
                call_id=call_id, name=name, arguments=json.dumps(arguments)
            )
        ],
        role="assistant",
        author_name=author,
    )


class FakeChatClient(FunctionInvocationLayer, ChatMiddlewareLayer, BaseChatClient):
    """Plays back scripted update sequences; records every model call.

    Each entry in `scripts` is the list of ChatResponseUpdate objects for
    one model call. When the function-invocation layer feeds tool results
    back, that next model call consumes the next script entry.
    """

    def __init__(self, scripts: Sequence[Sequence[ChatResponseUpdate]] = (), **kwargs: Any):
        super().__init__(**kwargs)
        self.scripts = [list(s) for s in scripts]
        self.calls: list[dict[str, Any]] = []

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ):
        self.calls.append(
            {
                "messages": list(messages),
                "options": dict(options),
                "stream": stream,
            }
        )
        if not self.scripts:
            raise AssertionError(
                "FakeChatClient ran out of scripted responses "
                f"(call #{len(self.calls)})"
            )
        updates = self.scripts.pop(0)

        if stream:
            async def gen():
                for u in updates:
                    yield u

            return ResponseStream(gen(), finalizer=ChatResponse.from_updates)

        async def _non_stream() -> ChatResponse:
            return ChatResponse.from_updates(updates)

        return _non_stream()

    # --- introspection helpers for assertions -------------------------

    def user_messages(self, call_index: int) -> list[str]:
        """Texts of user-role messages the model saw on a given call."""
        return [
            m.text
            for m in self.calls[call_index]["messages"]
            if getattr(m.role, "value", m.role) == "user"
        ]

    def instructions(self, call_index: int):
        return self.calls[call_index]["options"].get("instructions")
