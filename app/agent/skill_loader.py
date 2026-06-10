"""SkillLoader — filesystem-based skill discovery with progressive disclosure.

Layout (per HOW_SKILL_WORK.md):

  skills/
  └── my-skill/
      ├── SKILL.md          ← frontmatter (name, description) + instructions
      └── *.md / scripts/   ← resources loaded on demand via read tool

Loading levels:
  1. Startup  — name + description injected into system prompt
  2. Triggered — full SKILL.md body loaded when skill is invoked
  3. On demand — extra resources read by the agent via the read tool
"""
from __future__ import annotations

import re
from pathlib import Path


class Skill:
    def __init__(self, name: str, description: str, body: str, path: Path) -> None:
        self.name = name
        self.description = description
        self.body = body
        self.path = path          # path to SKILL.md


class SkillLoader:
    """Discovers skills under a directory and exposes them to the agent."""

    def __init__(self, skills_dir: str | Path = "skills") -> None:
        self._dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        if not self._dir.exists():
            return
        for skill_file in sorted(self._dir.rglob("SKILL.md")):
            skill = self._parse(skill_file)
            if skill:
                self._skills[skill.name] = skill

    def _parse(self, path: Path) -> Skill | None:
        text = path.read_text(encoding="utf-8")
        match = re.match(r"^---\n(.*?)\n---\n?(.*)", text, re.DOTALL)
        meta: dict[str, str] = {}
        body = text
        if match:
            for line in match.group(1).strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = match.group(2).strip()

        name = meta.get("name", path.parent.name)
        description = meta.get("description", "")

        # Validate per spec
        if not name or len(name) > 64:
            return None
        if not re.match(r"^[a-z0-9-]+$", name):
            return None
        if not description:
            return None

        return Skill(name=name, description=description, body=body, path=path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available(self) -> list[str]:
        return list(self._skills.keys())

    def system_prompt_section(self) -> str:
        """Level-1 content injected into every system prompt."""
        if not self._skills:
            return ""
        lines = ["## Available Skills", ""]
        for s in self._skills.values():
            lines.append(f"- **{s.name}**: {s.description}")
        lines.append(
            "\nTo use a skill, call the `load_skill` tool with the skill name. "
            "This loads the full instructions for that skill into your context."
        )
        return "\n".join(lines)

    def load(self, name: str) -> str:
        """Level-2 load: return the full SKILL.md body wrapped in a skill tag."""
        skill = self._skills.get(name)
        if skill is None:
            available = ", ".join(self._skills.keys()) or "(none)"
            return f"Error: unknown skill '{name}'. Available: {available}"
        return f'<skill name="{name}">\n{skill.body}\n</skill>'

    def skill_tool_schema(self) -> dict:
        """Return the Anthropic tool schema for load_skill."""
        names = self.available()
        schema: dict = {
            "name": "load_skill",
            "description": (
                "Load the full instructions for a named skill into the conversation context. "
                "Call this when the user's request matches a skill's description. "
                "The skill body contains step-by-step guidance and examples. "
                f"Available skills: {', '.join(names) if names else '(none)'}."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name to load.",
                        "enum": names if names else ["(none)"],
                    }
                },
                "required": ["name"],
            },
        }
        return schema

    def reload(self) -> None:
        """Re-scan the skills directory (useful for hot-reloading during development)."""
        self._skills.clear()
        self._load_all()
