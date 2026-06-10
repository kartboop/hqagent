"""TaskManager — file-backed task tracker with blocking relationships.

Tasks are stored as individual JSON files under .hqagent/tasks/.
Statuses: pending → in_progress → completed | deleted
"""
from __future__ import annotations

import json
from pathlib import Path

TASKS_DIR = Path(".hqagent/tasks")


class TaskManager:
    def __init__(self, tasks_dir: str | Path = TASKS_DIR) -> None:
        self._dir = Path(tasks_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        ids = [int(f.stem.split("_")[1]) for f in self._dir.glob("task_*.json")]
        return max(ids, default=0) + 1

    def _path(self, tid: int) -> Path:
        return self._dir / f"task_{tid}.json"

    def _load(self, tid: int) -> dict:
        p = self._path(tid)
        if not p.exists():
            raise ValueError(f"Task {tid} not found")
        return json.loads(p.read_text())

    def _save(self, task: dict) -> None:
        self._path(task["id"]).write_text(json.dumps(task, indent=2))

    def _all(self) -> list[dict]:
        return [
            json.loads(f.read_text())
            for f in sorted(self._dir.glob("task_*.json"))
        ]

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, subject: str, description: str = "") -> str:
        task = {
            "id": self._next_id(),
            "subject": subject,
            "description": description,
            "status": "pending",
            "owner": None,
            "blockedBy": [],
            "blocks": [],
        }
        self._save(task)
        return json.dumps(task, indent=2)

    def get(self, tid: int) -> str:
        return json.dumps(self._load(tid), indent=2)

    def update(
        self,
        tid: int,
        status: str | None = None,
        description: str | None = None,
        add_blocked_by: list[int] | None = None,
        add_blocks: list[int] | None = None,
    ) -> str:
        task = self._load(tid)

        if description is not None:
            task["description"] = description

        if status:
            task["status"] = status
            if status == "completed":
                # Unblock tasks that were waiting on this one
                for f in self._dir.glob("task_*.json"):
                    t = json.loads(f.read_text())
                    if tid in t.get("blockedBy", []):
                        t["blockedBy"].remove(tid)
                        self._save(t)
            if status == "deleted":
                self._path(tid).unlink(missing_ok=True)
                return f"Task {tid} deleted"

        if add_blocked_by:
            task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
        if add_blocks:
            task["blocks"] = list(set(task["blocks"] + add_blocks))

        self._save(task)
        return json.dumps(task, indent=2)

    def claim(self, tid: int, owner: str) -> str:
        task = self._load(tid)
        if task.get("blockedBy"):
            return f"Error: Task {tid} is blocked by {task['blockedBy']}"
        task["owner"] = owner
        task["status"] = "in_progress"
        self._save(task)
        return f"Claimed task #{tid} for {owner}"

    def list_all(self, status_filter: str | None = None) -> str:
        tasks = self._all()
        if status_filter:
            tasks = [t for t in tasks if t["status"] == status_filter]
        if not tasks:
            return "No tasks."
        icons = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]", "deleted": "[d]"}
        lines = []
        for t in tasks:
            icon = icons.get(t["status"], "[?]")
            owner = f" @{t['owner']}" if t.get("owner") else ""
            blocked = f" (blocked by: {t['blockedBy']})" if t.get("blockedBy") else ""
            lines.append(f"{icon} #{t['id']}: {t['subject']}{owner}{blocked}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool schemas (for injection into agent tools list)
    # ------------------------------------------------------------------

    def tool_schemas(self) -> list[dict]:
        return [
            {
                "name": "task_create",
                "description": (
                    "Create a new task with a subject and optional description. "
                    "Use to track goals, sub-goals, or work items during a session. "
                    "Returns the new task as JSON."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Short task title."},
                        "description": {"type": "string", "description": "Optional details."},
                    },
                    "required": ["subject"],
                },
            },
            {
                "name": "task_update",
                "description": (
                    "Update a task's status, description, or blocking relationships. "
                    "Valid statuses: pending, in_progress, completed, deleted. "
                    "Setting status=completed automatically unblocks dependent tasks."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "Task ID."},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "deleted"],
                        },
                        "description": {"type": "string"},
                        "add_blocked_by": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Task IDs that must complete before this one.",
                        },
                        "add_blocks": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Task IDs that this task blocks.",
                        },
                    },
                    "required": ["id"],
                },
            },
            {
                "name": "task_list",
                "description": (
                    "List all tasks, optionally filtered by status. "
                    "Returns a formatted summary of each task with its status, owner, and blockers."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status_filter": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "Only return tasks with this status (omit for all).",
                        }
                    },
                },
            },
            {
                "name": "task_get",
                "description": "Return full details for a single task by ID.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer", "description": "Task ID."}
                    },
                    "required": ["id"],
                },
            },
        ]

    async def dispatch(self, name: str, input_: dict) -> str:
        """Route a tool call to the appropriate TaskManager method."""
        if name == "task_create":
            return self.create(input_["subject"], input_.get("description", ""))
        if name == "task_update":
            return self.update(
                int(input_["id"]),
                status=input_.get("status"),
                description=input_.get("description"),
                add_blocked_by=input_.get("add_blocked_by"),
                add_blocks=input_.get("add_blocks"),
            )
        if name == "task_list":
            return self.list_all(input_.get("status_filter"))
        if name == "task_get":
            return self.get(int(input_["id"]))
        return f"Error: unknown task tool '{name}'"
