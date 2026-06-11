"""Entry point — run the interactive agent REPL."""
import asyncio
import os

from dotenv import load_dotenv

from app.agent import AgentLoop, ClientConfig


def main() -> None:
    load_dotenv()

    config = ClientConfig(
        model=os.environ.get("MODEL", "claude-sonnet-4-6"),
        api_key=os.environ.get("API_KEY", ""),
        base_url=os.environ.get("BASE_URL", ""),
        system_prompt=(
            "You are hqagent, an expert coding assistant. "
            "You help users by reading files, executing commands, editing code, and writing new files. "
            "Use bash for file discovery (ls, rg, find). "
            "Use read to examine files instead of cat or sed. "
            "Use edit for precise changes. "
            "Use write only for new files or complete rewrites."
        ),
        stream=True,
    )

    loop = AgentLoop(
        config,
        skills_dir="skills",
        tasks_dir=".hqagent/tasks",
        on_tool_call=lambda name, inp: print(f"\n[tool: {name}] ", end="", flush=True),
    )

    asyncio.run(loop.run_interactive())


if __name__ == "__main__":
    main()
