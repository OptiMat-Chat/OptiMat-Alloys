# Tests

One suite lives here.

```
tests/
└── unit/         100 offline tests — no API key, no Ollama, no GPU
```

Run everything:

```bash
python -m pytest tests/unit -q          # ~6 s
```

Or in a container, without touching your environment:

```bash
docker build -f Dockerfile.dev -t optimat-dev .
docker run --rm optimat-dev python -m pytest tests/unit -x -q
```

`pytest` and `pytest-asyncio` are dev-only — they are listed in `requirements.txt`
but deliberately **not** installed into the production image.

---

## Why these tests exist

They were written for the migration from AutoGen to Microsoft Agent Framework.
Their job is to prove the port did **not** change behaviour, so most of them
assert *parity* rather than correctness: the same system message, the same tool
schemas, the same termination rule, the same rate-limit pacing.

That makes them unusually strict. Several deliberately fail when you change text
that reaches the model — which is the point. A failure means "you changed
something the model sees; confirm that was intended", not necessarily "you broke
something".

---

## `tests/unit/`

| File | What it guards |
|---|---|
| **`test_tool_schemas.py`** | **Golden tool schemas.** Every `Annotated[T, "..."]` description, every default, and the required-parameter set for all seven agent tools. Fails if a parameter description changes by one byte. |
| **`test_agent_factory.py`** | Agent construction, and the **SHA-256-pinned system messages**. Fails if the Scientist prompt changes at all. |
| **`test_runner.py`** | The conversation runner: the `"?"`-termination loop, the streaming adapter, session and instructions plumbing. This is what replaced AutoGen's `RoundRobinGroupChat` + `TextMentionTermination("?")`. |
| **`test_model_factory.py`** | Client wiring — OpenRouter base URL, attribution headers, rate-limit middleware; Ollama through the daemon's OpenAI-compatible `/v1` endpoint, which serves both local and cloud models. |
| **`test_ollama_config.py`** | The Ollama model registry: every entry carries the fields the UI reads, the default model is one of them, and the VRAM / tool-calling filters actually filter. Covers helpers with no runtime callers, so nothing else would catch a regression. |
| **`test_middleware_delay.py`** | The OpenRouter pacing window (3 s, 20 req/min) runs before **every** model call, not once per conversation. |
| **`fake_chat_client.py`** | Not a test — a scripted client that subclasses the real layer stack minus telemetry, so `agent.run()` exercises genuine options merging, session history and the tool loop **with no network calls**. |

### Files that look deletable but are not

`tests/__init__.py` (a one-line docstring) and `tests/unit/__init__.py` (zero bytes)
are **package markers, not leftovers**. `test_runner.py` does

```python
from tests.unit.fake_chat_client import FakeChatClient
```

which only resolves if both directories are importable packages. Delete either and
collection aborts before a single test runs:

```
ERROR tests/unit/test_runner.py
Interrupted: 1 error during collection
```

They also give test modules fully-qualified names (`tests.unit.test_runner`), so two
files sharing a basename in different directories cannot collide.

`conftest.py` is the exception: it re-inserts the repository root on `sys.path`, which
the package markers already achieve, so the suite passes without it — including when
pytest is invoked from another directory. It is kept as a cheap safeguard against that
setup changing, not because anything currently depends on it.

### When a golden test fails

**`test_tool_param_descriptions_are_byte_identical`** — a tool's `Annotated`
description changed. If that was deliberate (fixing a misleading annotation, say),
update the map in `test_tool_schemas.py` and explain why in the commit message.
Regenerate the value from the live annotation rather than retyping it.

**`test_scientist_message_byte_identical`** — the system prompt changed. Same rule:
update `SCIENTIST_SHA256` only alongside a deliberate, reviewed prompt change.

**`test_attribution_headers_unchanged`** — the OpenRouter `HTTP-Referer` changed.
Note this header is intentionally **different** between the development repository
and this one; it is an attribution field used for OpenRouter's rankings.

---

## No browser suite

There are no end-to-end browser tests. An earlier Playwright suite covering the
first-run onboarding flow was removed in `d11d023` ("final release cleanup"),
along with its test report and manual-testing guide. It targeted an
authentication flow that no longer exists — it prompted for `OPENAI_API_KEY`,
which the application never reads; the keys in use are `OLLAMA_API_KEY` and
`OPENROUTER_API_KEY`.

To read it before writing a replacement:

```bash
git show d11d023^:tests/playwright/test_fresh_clone_experience.py
git show d11d023^:tests/playwright/MANUAL_TESTING.md
```
