"""Core agent tools: read, bash, edit, write.

Each tool is registered as a ToolDef (Anthropic tool schema) and has a
matching async handler function.  The ToolRegistry maps name → (schema, handler).
"""
from __future__ import annotations

import asyncio
import base64
import difflib
import mimetypes
import os
import re
import signal
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# ──────────────────────────────────────────────
# Limits (matching the spec in 需要实现的核心工具.md)
# ──────────────────────────────────────────────
READ_MAX_LINES = 2000
READ_MAX_BYTES = 50 * 1024          # 50 KB
BASH_MAX_LINES = 2000
BASH_MAX_BYTES = 50 * 1024
BASH_DEFAULT_TIMEOUT = 30           # seconds
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
IMAGE_MAX_PX = 1568                 # resize long-edge to this
TEMP_DIR = Path(tempfile.gettempdir())


# ──────────────────────────────────────────────
# Tool registry
# ──────────────────────────────────────────────

@dataclass
class ToolDef:
    schema: dict
    handler: Callable[..., Any]   # async (input_dict) -> str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.schema["name"]] = tool

    def schemas(self) -> list[dict]:
        return [t.schema for t in self._tools.values()]

    async def dispatch(self, name: str, input_: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'"
        try:
            return await tool.handler(input_)
        except Exception as exc:
            return f"Error: {exc}"

    def names(self) -> list[str]:
        return list(self._tools.keys())


# ──────────────────────────────────────────────
# Helper: truncate output
# ──────────────────────────────────────────────

def _truncate(text: str, max_lines: int, max_bytes: int, tmp_label: str = "") -> str:
    lines = text.splitlines(keepends=True)
    byte_count = 0
    cut_line = len(lines)
    for i, line in enumerate(lines):
        byte_count += len(line.encode())
        if i + 1 >= max_lines or byte_count >= max_bytes:
            cut_line = i + 1
            break
    truncated = lines[:cut_line]
    result = "".join(truncated)
    if cut_line < len(lines):
        remaining = len(lines) - cut_line
        result += f"\n[... {remaining} more lines truncated"
        if tmp_label:
            result += f". Full output in {tmp_label}"
        result += "]"
    return result


# ──────────────────────────────────────────────
# read
# ──────────────────────────────────────────────

READ_SCHEMA = {
    "name": "read",
    "description": (
        "Read the contents of a file at a given path. For text files, returns lines with "
        "1-based line numbers. Supports pagination via offset/limit to read large files in "
        "chunks — use offset=N to continue reading after a truncation hint. For images "
        "(jpg, jpeg, png, gif, webp) returns the image as a base64-encoded attachment. "
        "Output is capped at 2000 lines / 50 KB; a continuation hint is appended when "
        "truncation occurs. Prefer read over cat or sed for examining files."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or relative path to the file."},
            "offset": {
                "type": "integer",
                "description": "1-based line number to start reading from (default: 1).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return (default: 2000).",
            },
        },
        "required": ["path"],
    },
}


async def _read_handler(inp: dict) -> str | list:
    path = Path(inp["path"])
    if not path.exists():
        return f"Error: file not found: {path}"

    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return _read_image(path)

    offset = max(1, int(inp.get("offset", 1)))
    limit = min(int(inp.get("limit", READ_MAX_LINES)), READ_MAX_LINES)

    try:
        raw = path.read_bytes()
    except OSError as e:
        return f"Error: {e}"

    text = raw.decode("utf-8", errors="replace")
    all_lines = text.splitlines(keepends=True)
    total = len(all_lines)
    start = offset - 1           # convert to 0-based
    end = min(start + limit, total)
    selected = all_lines[start:end]

    # Apply byte cap within the selected window
    byte_count = 0
    final_lines = []
    for line in selected:
        byte_count += len(line.encode())
        if byte_count > READ_MAX_BYTES:
            final_lines.append(f"[... byte limit reached]\n")
            break
        final_lines.append(line)

    numbered = "".join(
        f"{start + i + 1}\t{line}" for i, line in enumerate(final_lines)
    )

    tail = end
    if tail < total:
        numbered += f"\nUse offset={tail + 1} to continue reading ({total - tail} lines remaining)."

    return numbered


def _read_image(path: Path) -> list:
    """Return a content block list with an image attachment."""
    data = path.read_bytes()
    # Resize if PIL is available; otherwise pass through
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if max(w, h) > IMAGE_MAX_PX:
            ratio = IMAGE_MAX_PX / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            fmt = img.format or "PNG"
            img.save(buf, format=fmt)
            data = buf.getvalue()
    except ImportError:
        pass

    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    b64 = base64.standard_b64encode(data).decode()
    return [
        {"type": "text", "text": f"Image: {path}"},
        {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
    ]


# ──────────────────────────────────────────────
# bash
# ──────────────────────────────────────────────

BASH_SCHEMA = {
    "name": "bash",
    "description": (
        "Execute a shell command in bash and return the combined stdout/stderr output. "
        "Use for file operations (ls, grep, find, rg), running scripts, installing packages, "
        "and any other shell tasks. Output is capped at 2000 lines / 50 KB; when truncated "
        "the full output is written to a temp file whose path is included. "
        "The timeout parameter (default 30 s) kills the entire process tree on expiry. "
        "Do NOT use for reading files — use the read tool instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute."},
            "timeout": {
                "type": "integer",
                "description": "Seconds before the command is killed (default 30, max 600).",
            },
        },
        "required": ["command"],
    },
}


async def _bash_handler(inp: dict) -> str:
    command = inp["command"]
    timeout = min(int(inp.get("timeout", BASH_DEFAULT_TIMEOUT)), 600)

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=os.environ.copy(),
        start_new_session=True,     # detached process group for clean kill
    )

    chunks: list[bytes] = []
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        chunks.append(stdout)
        rc = proc.returncode
    except asyncio.TimeoutError:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()
        return f"Error: command timed out after {timeout}s"

    output = b"".join(chunks).decode("utf-8", errors="replace")

    # Write full output to temp if needed
    tmp_label = ""
    lines = output.splitlines()
    if len(lines) > BASH_MAX_LINES or len(output.encode()) > BASH_MAX_BYTES:
        tmp = TEMP_DIR / f"bash_output_{proc.pid}.txt"
        tmp.write_text(output)
        tmp_label = str(tmp)

    result = _truncate(output, BASH_MAX_LINES, BASH_MAX_BYTES, tmp_label)

    if rc != 0:
        result += f"\n[exit code: {rc}]"
    return result


# ──────────────────────────────────────────────
# edit
# ──────────────────────────────────────────────

EDIT_SCHEMA = {
    "name": "edit",
    "description": (
        "Make precise edits to an existing file by replacing exact text snippets. "
        "Each entry in edits[] specifies oldText (the text to find) and newText (its replacement). "
        "oldText is matched using fuzzy matching that tolerates minor whitespace/indentation "
        "differences, but must be unique within the file. "
        "All edits are applied against the ORIGINAL file content in parallel — earlier edits "
        "do NOT shift the offsets of later edits, so you can safely include multiple "
        "non-overlapping edits in a single call. "
        "Use one edit call with multiple edits[] entries rather than multiple sequential calls "
        "when editing several locations in the same file. "
        "Keep oldText as small as possible while still being unique."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit."},
            "edits": {
                "type": "array",
                "description": "List of text replacements to apply.",
                "items": {
                    "type": "object",
                    "properties": {
                        "oldText": {"type": "string", "description": "Exact text to replace (unique in file)."},
                        "newText": {"type": "string", "description": "Replacement text."},
                    },
                    "required": ["oldText", "newText"],
                },
            },
        },
        "required": ["path", "edits"],
    },
}


def _normalize(text: str) -> str:
    """Collapse runs of whitespace for fuzzy comparison."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n", "\n", text)
    return text.strip()


def _fuzzy_find(haystack: str, needle: str) -> tuple[int, int] | None:
    """Return (start, end) byte offsets of needle in haystack using fuzzy match.

    Tries exact match first, then normalised match.
    Returns None if not found or ambiguous.
    """
    # Exact
    idx = haystack.find(needle)
    if idx != -1 and haystack.find(needle, idx + 1) == -1:
        return idx, idx + len(needle)
    if idx == -1:
        # Normalised
        norm_haystack = _normalize(haystack)
        norm_needle = _normalize(needle)
        norm_idx = norm_haystack.find(norm_needle)
        if norm_idx == -1:
            return None
        # Map back to original — find via sequence matcher
        matcher = difflib.SequenceMatcher(None, haystack, needle, autojunk=False)
        best = matcher.find_longest_match(0, len(haystack), 0, len(needle))
        if best.size < len(needle) * 0.8:
            return None
        # Expand around best match
        start = best.a
        end = best.a + best.size
        return start, end
    # Ambiguous exact match
    return None


async def _edit_handler(inp: dict) -> str:
    path = Path(inp["path"])
    if not path.exists():
        return f"Error: file not found: {path}"

    edits = inp.get("edits", [])
    if not edits:
        return "Error: edits list is empty"

    raw = path.read_bytes()
    # Detect line endings
    crlf = b"\r\n" in raw
    content = raw.decode("utf-8", errors="replace")
    # Normalise to LF for processing
    working = content.replace("\r\n", "\n")

    results: list[tuple[int, int, str]] = []  # (start, end, newText)
    for edit in edits:
        old = edit["oldText"].replace("\r\n", "\n")
        new = edit["newText"].replace("\r\n", "\n")
        span = _fuzzy_find(working, old)
        if span is None:
            return f"Error: oldText not found (or ambiguous) in {path}:\n{old!r}"
        results.append((span[0], span[1], new))

    # Check for overlaps
    results.sort(key=lambda x: x[0])
    for i in range(len(results) - 1):
        if results[i][1] > results[i + 1][0]:
            return "Error: edits overlap — ensure oldText snippets are non-overlapping"

    # Apply in reverse order to keep offsets valid
    for start, end, new in reversed(results):
        working = working[:start] + new + working[end:]

    if crlf:
        working = working.replace("\n", "\r\n")

    path.write_bytes(working.encode("utf-8"))
    return f"Edited {path} ({len(edits)} change(s))"


# ──────────────────────────────────────────────
# write
# ──────────────────────────────────────────────

WRITE_SCHEMA = {
    "name": "write",
    "description": (
        "Create a new file or completely overwrite an existing file with the given content. "
        "Parent directories are created automatically. Use write only for new files or full "
        "rewrites — for targeted changes to existing files, use the edit tool instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path of the file to create or overwrite."},
            "content": {"type": "string", "description": "Full file content to write."},
        },
        "required": ["path", "content"],
    },
}


async def _write_handler(inp: dict) -> str:
    path = Path(inp["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(inp["content"], encoding="utf-8")
    lines = inp["content"].count("\n") + 1
    return f"Wrote {path} ({lines} lines)"


# ──────────────────────────────────────────────
# Build the default registry
# ──────────────────────────────────────────────

def build_core_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolDef(schema=READ_SCHEMA, handler=_read_handler))
    registry.register(ToolDef(schema=BASH_SCHEMA, handler=_bash_handler))
    registry.register(ToolDef(schema=EDIT_SCHEMA, handler=_edit_handler))
    registry.register(ToolDef(schema=WRITE_SCHEMA, handler=_write_handler))
    return registry
