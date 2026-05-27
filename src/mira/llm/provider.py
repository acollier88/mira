"""OpenRouter API provider with retry/fallback and tool calling support."""

from __future__ import annotations

import json
import logging
import os

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from mira.config import LLMConfig
from mira.exceptions import LLMError

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _is_openrouter(base_url: str) -> bool:
    """OpenRouter-specific behavior (model prefix stripping, ranking headers)
    is gated on the configured base_url. Any other URL (vLLM, Ollama,
    LiteLLM proxy, LocalAI, Together, Fireworks, Groq, etc.) gets the
    portable OpenAI-compatible request shape."""
    return base_url.rstrip("/") == _OPENROUTER_BASE_URL.rstrip("/")


# ---------------------------------------------------------------------------
# Tool schemas for structured output via function/tool calling
# ---------------------------------------------------------------------------

SUBMIT_REVIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_review",
        "description": "Submit your code review findings including comments, key issues, and a summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "comments": {
                    "type": "array",
                    "description": "List of review comments on specific lines of code.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative file path."},
                            "line": {
                                "type": "integer",
                                "description": "Line number in the target file.",
                            },
                            "end_line": {
                                "type": ["integer", "null"],
                                "description": "End line for multi-line comments, or null.",
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["blocker", "warning", "suggestion", "nitpick"],
                            },
                            "category": {
                                "type": "string",
                                "enum": [
                                    "bug",
                                    "security",
                                    "performance",
                                    "error-handling",
                                    "race-condition",
                                    "resource-leak",
                                    "maintainability",
                                    "clarity",
                                    "configuration",
                                    "other",
                                ],
                            },
                            "title": {"type": "string", "description": "Short title (<80 chars)."},
                            "body": {
                                "type": "string",
                                "description": "Detailed explanation of the issue. Use single backticks for inline code references. Do NOT use triple-backtick code blocks.",
                            },
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "existing_code": {
                                "type": "string",
                                "description": "Verbatim copy of the code from the diff that this comment targets. Must be an exact substring.",
                            },
                            "suggestion": {
                                "type": ["string", "null"],
                                "description": "Optional replacement code to fix the issue. Raw code only — do NOT wrap in backticks or markdown fences.",
                            },
                            "agent_prompt": {
                                "type": ["string", "null"],
                                "description": "Concise imperative instruction for AI coding agents.",
                            },
                        },
                        "required": [
                            "path",
                            "line",
                            "severity",
                            "category",
                            "title",
                            "body",
                            "confidence",
                            "existing_code",
                        ],
                    },
                },
                "key_issues": {
                    "type": "array",
                    "description": "1-3 most critical findings a human reviewer MUST examine.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue": {"type": "string"},
                            "path": {"type": "string"},
                            "line": {"type": "integer"},
                        },
                        "required": ["issue", "path", "line"],
                    },
                },
                "summary": {
                    "type": "string",
                    "description": "Brief overall summary of the review.",
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "reviewed_files": {"type": "integer"},
                        "skipped_reason": {"type": ["string", "null"]},
                    },
                },
            },
            "required": ["comments", "summary"],
        },
    },
}

SUBMIT_CRITIQUE_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_critique",
        "description": (
            "For each draft review comment, decide whether it's a real, "
            "verifiable issue. Return the indices of comments worth keeping "
            "and a brief reason for each rejection."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "description": "One verdict per draft comment, in input order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "Zero-based index of the draft comment.",
                            },
                            "keep": {
                                "type": "boolean",
                                "description": (
                                    "true if the comment cites specific code that proves "
                                    "the issue, the reasoning is correct, and the fix is "
                                    "actionable. false for confident-but-wrong claims, "
                                    "speculation, or 'while I'm here' style nits."
                                ),
                            },
                            "reason": {
                                "type": "string",
                                "description": "One short sentence explaining the verdict.",
                            },
                        },
                        "required": ["index", "keep", "reason"],
                    },
                },
            },
            "required": ["verdicts"],
        },
    },
}


SUBMIT_THREAD_REPLY_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_thread_reply",
        "description": (
            "Reply to a human's comment on one of your previous PR review "
            "suggestions. Classify their intent and write a short reply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["disagreement", "question", "agreement", "other"],
                    "description": (
                        "disagreement = human refutes the suggestion / says it doesn't apply. "
                        "question = human is asking for clarification. "
                        "agreement = human is acknowledging or thanking. "
                        "other = anything else (off-topic, unclear)."
                    ),
                },
                "reply": {
                    "type": "string",
                    "description": (
                        "Your reply, 1-2 short sentences, plain text, no markdown. "
                        'No emojis, no apologies, no "as an AI". For disagreement, '
                        "concede gracefully. For questions, answer directly."
                    ),
                },
            },
            "required": ["intent", "reply"],
        },
    },
}


SUBMIT_WALKTHROUGH_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_walkthrough",
        "description": "Submit a high-level walkthrough summary of the pull request.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Brief overall summary of the PR."},
                "confidence_score": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                        "label": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["score", "label", "reason"],
                },
                "change_groups": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "files": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"},
                                        "change_type": {
                                            "type": "string",
                                            "enum": ["added", "modified", "deleted", "renamed"],
                                        },
                                        "description": {"type": "string"},
                                    },
                                    "required": ["path", "change_type", "description"],
                                },
                            },
                        },
                        "required": ["label", "files"],
                    },
                },
                "effort": {
                    "type": "object",
                    "properties": {
                        "level": {"type": "integer", "minimum": 1, "maximum": 5},
                        "label": {"type": "string"},
                        "minutes": {"type": "integer"},
                    },
                    "required": ["level", "label", "minutes"],
                },
                "sequence_diagram": {
                    "type": ["string", "null"],
                    "description": "Mermaid sequence diagram or null.",
                },
            },
            "required": ["summary", "change_groups"],
        },
    },
}


def _get_api_key(config: LLMConfig) -> str:
    """Resolve the API key for the configured endpoint.

    Reads from `config.api_key_env` first, then falls back to the legacy
    `OPENROUTER_API_KEY` / `OPENAI_API_KEY` lookup for backward compatibility.
    If `api_key_env` is explicitly set to "" the empty string is returned
    without error — useful for local endpoints (Ollama, llama.cpp server)
    that don't require auth.
    """
    if config.api_key_env == "":
        # Explicit opt-out — local endpoint with no auth.
        return ""
    key = os.environ.get(config.api_key_env, "")
    if not key:
        # Back-compat fallback so existing OPENROUTER_API_KEY / OPENAI_API_KEY
        # setups keep working without changes.
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise LLMError(
            f"No API key found. Set {config.api_key_env} (or OPENROUTER_API_KEY / "
            f'OPENAI_API_KEY) in the environment, or set llm.api_key_env: "" in '
            f"your config for a local endpoint that needs no auth."
        )
    return key


def _strip_model_prefix(model: str, base_url: str) -> str:
    """Strip 'openrouter/' prefix only when targeting OpenRouter; other
    endpoints accept (and often require) the full model string."""
    if _is_openrouter(base_url) and model.startswith("openrouter/"):
        return model[len("openrouter/") :]
    return model


def _extract_error_message(resp: httpx.Response) -> str:
    """Best-effort extraction of a provider error message."""
    try:
        data = resp.json()
    except Exception:
        return resp.text

    if isinstance(data, dict):
        for key in ("error", "message", "detail"):
            value = data.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                nested = value.get("message") or value.get("detail")
                if isinstance(nested, str):
                    return nested
    return resp.text


def _error_kind(status_code: int, message: str) -> str | None:
    lower = message.lower()
    if status_code in (401, 403) or any(
        token in lower
        for token in ("unauthorized", "authentication", "invalid api key", "invalid_api_key")
    ):
        return "auth"
    if any(
        token in lower
        for token in ("response_format", "json_object", "json mode", "json schema")
    ):
        return "json_mode"
    if any(
        token in lower
        for token in (
            "tool_choice",
            "tool call",
            "tool_calls",
            "function calling",
            "function_call",
            "tools are not supported",
        )
    ):
        return "tool_calling"
    return None


def _format_api_error(status_code: int, message: str) -> str:
    kind = _error_kind(status_code, message)
    if kind == "auth":
        return f"LLM authentication failed ({status_code}): {message}"
    if kind == "json_mode":
        return f"LLM endpoint does not support JSON mode ({status_code}): {message}"
    if kind == "tool_calling":
        return f"LLM endpoint does not support tool calling ({status_code}): {message}"
    return f"LLM API error {status_code}: {message}"


def _force_json_messages(
    messages: list[dict[str, str]],
    schema_hint: dict | None = None,
) -> list[dict[str, str]]:
    """Append a portability fallback instruction for endpoints without structured-output support."""
    prompt = "Return ONLY a valid JSON object. Do not include markdown fences or extra prose."
    if schema_hint is not None:
        prompt += f" Match this JSON schema: {json.dumps(schema_hint, sort_keys=True)}"
    return [
        *messages,
        {
            "role": "system",
            "content": prompt,
        },
    ]


class LLMProvider:
    """Direct OpenRouter API client for LLM completions."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._supports_json_mode: bool | None = None
        self._supports_tool_calling: bool | None = None

    def _chat_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _build_headers(self) -> dict[str, str]:
        """Build request headers. OpenRouter-specific ranking headers are
        only attached when targeting OpenRouter; other endpoints get a clean
        portable header set. Authorization is omitted entirely if the
        endpoint needs no key (Ollama, llama.cpp, etc.)."""
        if hasattr(self, "_cached_headers"):
            return dict(self._cached_headers)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = _get_api_key(self.config)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if _is_openrouter(self.config.base_url):
            headers["HTTP-Referer"] = "https://github.com/miracodeai/mira"
            headers["X-Title"] = "Mira Code Reviewer"
        self._cached_headers = headers
        return dict(headers)

    async def _post_chat(self, body: dict) -> httpx.Response:
        async with httpx.AsyncClient(timeout=120) as client:
            return await client.post(
                self._chat_url(),
                headers=self._build_headers(),
                json=body,
            )

    def _track_usage(self, data: dict) -> None:
        usage = data.get("usage")
        if usage:
            self.total_prompt_tokens += usage.get("prompt_tokens", 0)
            self.total_completion_tokens += usage.get("completion_tokens", 0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call_llm(
        self,
        model: str,
        messages: list[dict[str, str]],
        json_mode: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Make a single LLM call with retries against the configured endpoint."""
        request_messages = messages
        body: dict = {
            "model": _strip_model_prefix(model, self.config.base_url),
            "messages": request_messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        if json_mode and self._supports_json_mode is False:
            request_messages = _force_json_messages(messages)
            body["messages"] = request_messages
        elif json_mode:
            body["response_format"] = {"type": "json_object"}

        resp = await self._post_chat(body)
        if resp.status_code != 200:
            message = _extract_error_message(resp)
            if json_mode and _error_kind(resp.status_code, message) == "json_mode":
                logger.warning(
                    "Endpoint %s rejected JSON mode; retrying with prompt-only JSON fallback",
                    self.config.base_url,
                )
                self._supports_json_mode = False
                fallback_body = dict(body)
                fallback_body.pop("response_format", None)
                fallback_body["messages"] = _force_json_messages(messages)
                resp = await self._post_chat(fallback_body)
                if resp.status_code != 200:
                    raise LLMError(_format_api_error(resp.status_code, _extract_error_message(resp)))
            else:
                raise LLMError(_format_api_error(resp.status_code, message))
        else:
            if json_mode and self._supports_json_mode is None:
                self._supports_json_mode = True
        data = resp.json()

        content = data["choices"][0]["message"].get("content") or ""
        self._track_usage(data)

        return content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call_llm_with_tools(
        self,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float | None = None,
    ) -> str:
        """Make an LLM call with tool/function calling and retries.

        The LLM returns structured data by 'calling' a tool. We extract the
        tool arguments as the JSON response.
        """
        request_messages = messages
        body: dict = {
            "model": _strip_model_prefix(model, self.config.base_url),
            "messages": request_messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self._supports_tool_calling is False:
            request_messages = _force_json_messages(messages, schema_hint=tools[0]["function"]["parameters"])
            body["messages"] = request_messages
        else:
            body["tools"] = tools
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": tools[0]["function"]["name"]},
            }

        resp = await self._post_chat(body)
        if resp.status_code != 200:
            message = _extract_error_message(resp)
            if _error_kind(resp.status_code, message) == "tool_calling":
                logger.warning(
                    "Endpoint %s rejected tool calling; retrying with prompt-only JSON fallback",
                    self.config.base_url,
                )
                self._supports_tool_calling = False
                fallback_body = {
                    "model": _strip_model_prefix(model, self.config.base_url),
                    "messages": _force_json_messages(
                        messages,
                        schema_hint=tools[0]["function"]["parameters"],
                    ),
                    "temperature": temperature if temperature is not None else self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
                resp = await self._post_chat(fallback_body)
                if resp.status_code != 200:
                    raise LLMError(_format_api_error(resp.status_code, _extract_error_message(resp)))
            else:
                raise LLMError(_format_api_error(resp.status_code, message))
        else:
            if self._supports_tool_calling is None and "tools" in body:
                self._supports_tool_calling = True
        data = resp.json()
        self._track_usage(data)

        # Extract tool call arguments
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls")

        if tool_calls and len(tool_calls) > 0:
            return tool_calls[0]["function"]["arguments"]

        # Fallback: if the model returned content instead of a tool call,
        # return the content as-is (some models may accept tools but answer
        # with raw JSON content instead of a formal tool call envelope).
        content = message.get("content") or ""
        if content:
            self._supports_tool_calling = False
            logger.warning("Model returned content instead of tool call, using content as fallback")
            return content

        raise LLMError(
            "LLM endpoint returned neither tool calls nor JSON content; verify OpenAI-compatible chat-completions support."
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Complete a prompt using JSON mode, with fallback model support.

        Args:
            temperature: Override the default temperature for this call.
                         Use ``0.0`` for deterministic tasks like verification.
            max_tokens: Override the default output token cap for this call.
                        Indexing summarization needs ~16k to avoid truncation
                        on large batches; the default 4096 cuts JSON off.
        """
        try:
            return await self._call_llm(
                self.config.model,
                messages,
                json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as primary_err:
            if self.config.fallback_model:
                logger.warning(
                    "Primary model %s failed (%s), trying fallback %s",
                    self.config.model,
                    primary_err,
                    self.config.fallback_model,
                )
                try:
                    return await self._call_llm(
                        self.config.fallback_model,
                        messages,
                        json_mode,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as fallback_err:
                    raise LLMError(
                        f"Both primary ({self.config.model}) and fallback "
                        f"({self.config.fallback_model}) models failed: {fallback_err}"
                    ) from fallback_err
            raise LLMError(
                f"LLM completion failed with {self.config.model}: {primary_err}"
            ) from primary_err

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def _call_llm_agentic(
        self,
        model: str,
        messages: list,
        tools: list[dict],
        temperature: float | None = None,
    ) -> dict:
        """Make a tool-using LLM call without forcing a specific tool.

        Unlike `_call_llm_with_tools`, this returns the *full* assistant
        message (with `tool_calls` and `content`) so the caller can
        dispatch the calls and continue the conversation. This is what
        the agentic loop needs.
        """
        body: dict = {
            "model": _strip_model_prefix(model, self.config.base_url),
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        resp = await self._post_chat(body)
        if resp.status_code != 200:
            raise LLMError(_format_api_error(resp.status_code, _extract_error_message(resp)))
        data = resp.json()
        self._track_usage(data)

        return data["choices"][0]["message"]

    async def complete_agentic(
        self,
        messages: list,
        tools: list[dict],
        temperature: float | None = None,
    ) -> dict:
        """Single hop of an agentic loop. Returns the assistant message dict.

        The caller is responsible for the loop: append the message,
        dispatch any `tool_calls`, append the tool results as `tool`-role
        messages, and call again until the terminal tool fires.
        """
        try:
            return await self._call_llm_agentic(
                self.config.model, messages, tools, temperature=temperature
            )
        except Exception as primary_err:
            if self.config.fallback_model:
                logger.warning(
                    "Primary model %s failed (%s), trying fallback %s",
                    self.config.model,
                    primary_err,
                    self.config.fallback_model,
                )
                try:
                    return await self._call_llm_agentic(
                        self.config.fallback_model, messages, tools, temperature=temperature
                    )
                except Exception as fallback_err:
                    raise LLMError(
                        f"Both primary ({self.config.model}) and fallback "
                        f"({self.config.fallback_model}) models failed: {fallback_err}"
                    ) from fallback_err
            raise LLMError(
                f"LLM agentic call failed with {self.config.model}: {primary_err}"
            ) from primary_err

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float | None = None,
    ) -> str:
        """Complete a prompt using tool calling for structured output.

        The LLM 'calls' a tool to return structured JSON data. Works reliably
        across all models available on OpenRouter.

        Args:
            messages: The prompt messages.
            tools: Tool schemas in OpenAI function-calling format.
            temperature: Override the default temperature.

        Returns:
            The JSON string from the tool call arguments.
        """
        try:
            return await self._call_llm_with_tools(
                self.config.model, messages, tools, temperature=temperature
            )
        except Exception as primary_err:
            if self.config.fallback_model:
                logger.warning(
                    "Primary model %s failed (%s), trying fallback %s",
                    self.config.model,
                    primary_err,
                    self.config.fallback_model,
                )
                try:
                    return await self._call_llm_with_tools(
                        self.config.fallback_model, messages, tools, temperature=temperature
                    )
                except Exception as fallback_err:
                    raise LLMError(
                        f"Both primary ({self.config.model}) and fallback "
                        f"({self.config.fallback_model}) models failed: {fallback_err}"
                    ) from fallback_err
            raise LLMError(
                f"LLM tool-call failed with {self.config.model}: {primary_err}"
            ) from primary_err

    async def review(self, messages: list[dict[str, str]]) -> str:
        """Submit a review using tool calling.

        Returns the JSON string containing review comments, key issues, and summary.
        """
        return await self.complete_with_tools(messages, tools=[SUBMIT_REVIEW_TOOL])

    async def walkthrough(self, messages: list[dict[str, str]]) -> str:
        """Submit a walkthrough using tool calling.

        Returns the JSON string containing walkthrough summary and file changes.
        """
        return await self.complete_with_tools(messages, tools=[SUBMIT_WALKTHROUGH_TOOL])

    def count_tokens(self, text: str) -> int:
        """Estimate token count. Uses ~4 chars per token heuristic."""
        return len(text) // 4

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
