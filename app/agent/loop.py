"""Agent loop — orchestrates LLM ↔ tool calls until end_turn or user interrupt.

Flow:
  1. Build system prompt (base + skills section)
  2. Call LLM with current context + tool schemas
  3. Handle stop_reason:
     - tool_use   → dispatch tools, append results, loop
     - max_tokens → append continuation prompt, loop
     - pause_turn → append assistant turn, loop
     - end_turn   → done
     - refusal    → surface to caller
  4. After each round, check context size and compact if needed
"""
from __future__ import annotations

import asyncio
from typing import Callable

from .context import Context
from .llm import ClientConfig, LLMClient
from .skill_loader import SkillLoader
from .task_manager import TaskManager
from .tools import ToolRegistry, build_core_registry

# How many max_tokens continuations to allow before giving up
MAX_TOKEN_CONTINUATIONS = 5
# How many pause_turn continuations to allow
MAX_PAUSE_CONTINUATIONS = 10


def _build_system_prompt(base: str, skill_loader: SkillLoader | None) -> str:
    parts = [base.strip()]
    if skill_loader:
        section = skill_loader.system_prompt_section()
        if section:
            parts.append(section)
    parts.append(
        "## Tools\n"
        "- `read`: Read file contents (prefer over cat/sed)\n"
        "- `bash`: Execute shell commands\n"
        "- `edit`: Precise file edits via oldText→newText replacement\n"
        "- `write`: Create or overwrite files\n"
        "- `load_skill`: Load full instructions for a named skill\n"
        "- `task_create/update/list/get`: Manage work-item tasks\n"
    )
    return "\n\n".join(parts)


class AgentLoop:
    """Run the agent loop for a single session."""

    def __init__(
        self,
        config: ClientConfig,
        *,
        skills_dir: str = "skills",
        tasks_dir: str = ".hqagent/tasks",
        on_text: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, str], None] | None = None,
    ) -> None:
        self.llm = LLMClient(config)
        self.context = Context()
        self.registry: ToolRegistry = build_core_registry()
        self.skill_loader = SkillLoader(skills_dir)
        self.task_manager = TaskManager(tasks_dir)

        # Register skill + task tools into the registry
        self._register_skill_tool()
        self._register_task_tools()

        # Callbacks for streaming output / observability
        self.on_text = on_text or (lambda t: print(t, end="", flush=True))
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result

        # Patch system prompt
        self.llm.cfg.system_prompt = _build_system_prompt(
            self.llm.cfg.system_prompt, self.skill_loader
        )

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_skill_tool(self) -> None:
        from .tools import ToolDef

        schema = self.skill_loader.skill_tool_schema()

        async def handler(inp: dict) -> str:
            return self.skill_loader.load(inp["name"])

        self.registry.register(ToolDef(schema=schema, handler=handler))

    def _register_task_tools(self) -> None:
        from .tools import ToolDef

        for schema in self.task_manager.tool_schemas():
            name = schema["name"]

            async def handler(inp: dict, _name: str = name) -> str:
                return await self.task_manager.dispatch(_name, inp)

            self.registry.register(ToolDef(schema=schema, handler=handler))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, user_message: str) -> str:
        """Add user_message and run until the agent is done. Returns final text."""
        self.context.add_user(user_message)
        return await self._loop()

    async def run_interactive(self) -> None:
        """REPL: read user input from stdin, run loop, print output."""
        print("hqagent ready. Type your message (Ctrl-D to exit).\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if not user_input:
                continue
            print("Agent: ", end="")
            await self.run(user_input)
            print()  # newline after streaming output

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def _loop(self) -> str:
        token_continuations = 0
        pause_continuations = 0
        final_text = ""

        while True:
            # Auto-compact if context is large
            if self.context.should_compact():
                self.context.micro_compact()
                if self.context.should_compact():
                    await self.context.auto_compact(self.llm)

            resp = await self.llm.create(
                messages=self.context.to_list(),
                tools=self.registry.schemas(),
                on_text=self.on_text,
            )

            stop = resp.stop_reason

            # ── end_turn ──────────────────────────────────────────────
            if stop == "end_turn":
                text = self.llm.extract_text(resp)
                if not text and not resp.content:
                    # Empty end_turn — add continuation nudge (once)
                    self.context.add_assistant(resp.content)
                    self.context.add_user("Please continue.")
                    continue
                self.context.add_assistant(resp.content)
                final_text = text
                break

            # ── tool_use ──────────────────────────────────────────────
            if stop == "tool_use":
                self.context.add_assistant(resp.content)
                tool_uses = self.llm.extract_tool_uses(resp)
                results = await self._dispatch_tools(tool_uses)
                self.context.add_tool_results(results)
                continue

            # ── max_tokens — truncated response ───────────────────────
            if stop == "max_tokens":
                token_continuations += 1
                if token_continuations > MAX_TOKEN_CONTINUATIONS:
                    text = self.llm.extract_text(resp)
                    text += "\n\n[Response truncated: max continuation limit reached]"
                    self.context.add_assistant(resp.content)
                    final_text = text
                    break
                # Append partial and ask Claude to continue
                self.context.add_assistant(resp.content)
                self.context.add_user("Please continue from where you left off.")
                continue

            # ── pause_turn — server-side tool loop iteration limit ─────
            if stop == "pause_turn":
                pause_continuations += 1
                if pause_continuations > MAX_PAUSE_CONTINUATIONS:
                    self.context.add_assistant(resp.content)
                    final_text = self.llm.extract_text(resp)
                    break
                self.context.add_assistant(resp.content)
                continue

            # ── refusal ───────────────────────────────────────────────
            if stop == "refusal":
                text = self.llm.extract_text(resp) or "[Refused]"
                self.context.add_assistant(resp.content)
                final_text = text
                break

            # ── stop_sequence or unknown ──────────────────────────────
            text = self.llm.extract_text(resp)
            self.context.add_assistant(resp.content)
            final_text = text
            break

        return final_text

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def _dispatch_tools(self, tool_uses: list[dict]) -> list[dict]:
        """Run all tool calls (possibly in parallel) and return result blocks."""
        tasks = [self._run_one_tool(tu) for tu in tool_uses]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _run_one_tool(self, tool_use: dict) -> dict:
        name = tool_use["name"]
        inp = tool_use["input"]
        tool_id = tool_use["id"]

        if self.on_tool_call:
            self.on_tool_call(name, inp)

        result = await self.registry.dispatch(name, inp)

        if self.on_tool_result:
            self.on_tool_result(name, result)

        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": result if isinstance(result, str) else str(result),
        }
