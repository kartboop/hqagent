"""LLM client — wraps AsyncAnthropic with streaming, retry, and token tracking."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import anthropic
from anthropic import AsyncAnthropic


@dataclass
class ClientConfig:
    model: str = "claude-sonnet-4-6"
    api_key: str = ""
    base_url: str = ""
    system_prompt: str = "You are a helpful assistant."
    max_tokens: int = 16384
    max_retries: int = 3
    timeout: float = 120.0
    thinking: bool = False
    stream: bool = True
    extra_headers: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def update(self, usage_obj) -> None:
        if usage_obj is None:
            return
        self.input_tokens += getattr(usage_obj, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage_obj, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage_obj, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage_obj, "cache_creation_input_tokens", 0) or 0

    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient:
    """Async Anthropic client with streaming, retry, and per-session usage tracking."""

    def __init__(self, config: ClientConfig):
        self.cfg = config
        self.usage = Usage()
        kwargs: dict = {
            "api_key": config.api_key,
            "max_retries": config.max_retries,
            "timeout": config.timeout,
            "default_headers": config.extra_headers,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = AsyncAnthropic(**kwargs)

    def _build_tools_param(self, tools: list[dict]) -> list[dict] | None:
        return tools if tools else None

    def _extra_params(self) -> dict:
        params: dict = {}
        if self.cfg.thinking:
            params["thinking"] = {"type": "adaptive"}
        return params

    async def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> anthropic.types.Message:
        """Send a request. Streams if cfg.stream=True and on_text is provided.

        Returns the full Message object (content blocks already collected).
        Falls back to non-streaming on stream read errors.
        """
        tools_param = self._build_tools_param(tools or [])
        extra = self._extra_params()

        kwargs: dict = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=self.cfg.system_prompt,
            messages=messages,
            **extra,
        )
        if tools_param:
            kwargs["tools"] = tools_param

        if self.cfg.stream and on_text is not None:
            try:
                return await self._stream(kwargs, on_text, on_thinking)
            except Exception:
                return await self.create_non_stream(messages, tools=tools, on_text=on_text, on_thinking=on_thinking)
        else:
            resp = await self._client.messages.create(**kwargs)
            self.usage.update(resp.usage)
            return resp

    async def _stream(
        self,
        kwargs: dict,
        on_text: Callable[[str], None],
        on_thinking: Callable[[str], None] | None = None,
    ) -> anthropic.types.Message:
        current_block_type: str | None = None
        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    current_block_type = event.content_block.type
                    if current_block_type == "thinking" and on_thinking:
                        on_thinking("<think>\n")
                elif etype == "content_block_delta":
                    delta = event.delta
                    dtype = getattr(delta, "type", None)
                    if dtype == "thinking_delta" and on_thinking:
                        on_thinking(delta.thinking)
                    elif dtype == "text_delta":
                        on_text(delta.text)
                elif etype == "content_block_stop":
                    if current_block_type == "thinking" and on_thinking:
                        on_thinking("\n</think>\n\n")
                    current_block_type = None
            msg = await stream.get_final_message()
        self.usage.update(msg.usage)
        return msg

    async def create_non_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> anthropic.types.Message:
        """Non-streaming create that still calls on_text with the full response."""
        tools_param = self._build_tools_param(tools or [])
        extra = self._extra_params()

        kwargs: dict = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=self.cfg.system_prompt,
            messages=messages,
            **extra,
        )
        if tools_param:
            kwargs["tools"] = tools_param

        resp = await self._client.messages.create(**kwargs)
        self.usage.update(resp.usage)
        if on_thinking:
            for block in resp.content:
                if block.type == "thinking":
                    on_thinking(f"<think>\n{block.thinking}\n</think>\n\n")
        text = self.extract_text(resp)
        if on_text and text:
            on_text(text)
        return resp

    async def create_with_continuation(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
        max_continuations: int = 5,
        prefer_non_stream: bool = False,
    ) -> anthropic.types.Message:
        """Handle pause_turn by automatically continuing the conversation."""
        for _ in range(max_continuations):
            resp = await self.create(messages, tools=tools, on_text=on_text, on_thinking=on_thinking)
            if resp.stop_reason != "pause_turn":
                return resp
            # pause_turn: push assistant content and let Claude continue
            messages = list(messages) + [
                {"role": "assistant", "content": resp.content}
            ]
        return resp

    def extract_text(self, message: anthropic.types.Message) -> str:
        parts = []
        for block in message.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts)

    def extract_tool_uses(self, message: anthropic.types.Message) -> list[dict]:
        return [
            {"id": b.id, "name": b.name, "input": b.input}
            for b in message.content
            if b.type == "tool_use"
        ]
