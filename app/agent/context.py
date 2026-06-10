"""Conversation context — message history with compaction support."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient

TRANSCRIPTS_DIR = Path(".hqagent/transcripts")

# Compact when estimated input tokens exceed this threshold
COMPACT_THRESHOLD = 80_000


class Context:
    """Holds the mutable message list for one agent session.

    Provides helpers to append messages in the correct Anthropic format and
    to compress the history when it grows too large.
    """

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self._turn: int = 0

    # ------------------------------------------------------------------
    # Building messages
    # ------------------------------------------------------------------

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, content: list) -> None:
        """Append an assistant turn with a content block list (as returned by the API)."""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, results: list[dict]) -> None:
        """Append a user turn that contains only tool_result blocks.

        Per Anthropic best-practice: never mix text with tool_result in the
        same user message — doing so trains Claude to expect text after every
        tool use and causes empty end_turn responses.
        """
        self.messages.append({"role": "user", "content": results})

    def add_tool_result(self, tool_use_id: str, content: str, is_error: bool = False) -> None:
        """Convenience: append a single tool result as its own user turn."""
        block: dict = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            block["is_error"] = True
        self.add_tool_results([block])

    # ------------------------------------------------------------------
    # Token estimation (cheap, no API call needed)
    # ------------------------------------------------------------------

    def _estimate_tokens(self) -> int:
        """Rough token count: 1 token ≈ 4 chars in English/code."""
        raw = json.dumps(self.messages, default=str)
        return len(raw) // 4

    def should_compact(self) -> bool:
        return self._estimate_tokens() > COMPACT_THRESHOLD

    # ------------------------------------------------------------------
    # Micro-compaction: truncate large old tool results in-place
    # ------------------------------------------------------------------

    def micro_compact(self, keep_last: int = 3) -> None:
        """Clear oversized tool_result contents, keeping the N most recent."""
        tool_result_parts: list[dict] = []
        for msg in self.messages:
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "tool_result":
                        tool_result_parts.append(part)

        for part in tool_result_parts[:-keep_last]:
            content = part.get("content", "")
            if isinstance(content, str) and len(content) > 200:
                part["content"] = "[cleared — content truncated to save context]"

    # ------------------------------------------------------------------
    # Full compaction via LLM summarisation
    # ------------------------------------------------------------------

    async def auto_compact(self, llm: "LLMClient") -> None:
        """Summarise the current history and replace it with a compact stub."""
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = TRANSCRIPTS_DIR / f"transcript_{ts}.jsonl"
        with open(path, "w") as f:
            for msg in self.messages:
                f.write(json.dumps(msg, default=str) + "\n")

        # Summarise — use a fresh context so we don't recurse
        conv_text = json.dumps(self.messages, default=str)[:100_000]
        summary_messages = [
            {
                "role": "user",
                "content": (
                    "Summarise the following conversation for continuity. "
                    "Include: goals, decisions made, important findings, current state, "
                    "and any unfinished tasks. Be concise but complete.\n\n"
                    f"{conv_text}"
                ),
            }
        ]
        resp = await llm.create(summary_messages)
        summary = llm.extract_text(resp)

        self.messages = [
            {
                "role": "user",
                "content": (
                    f"[Context compressed. Full transcript saved to {path}]\n\n"
                    f"## Summary\n{summary}"
                ),
            },
            {
                "role": "assistant",
                "content": "Understood. I have the summary of our prior work and will continue from here.",
            },
        ]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_list(self) -> list[dict]:
        return self.messages

    def __len__(self) -> int:
        return len(self.messages)
