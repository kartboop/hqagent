"""FastAPI application exposing hqagent core capabilities as HTTP/SSE endpoints.

Run with:
    cd hqagent && uv run uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import os
import uuid
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent import AgentLoop, ClientConfig, SkillLoader, TaskManager, Usage

load_dotenv()

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

SESSION_TTL = 3600  # 1 hour idle timeout


class _Session:
    __slots__ = ("loop", "created_at", "last_active")

    def __init__(self, loop: AgentLoop) -> None:
        self.loop = loop
        self.created_at = time.time()
        self.last_active = time.time()


_sessions: dict[str, _Session] = {}


def _gc_sessions() -> int:
    """Remove expired sessions, return count of remaining."""
    now = time.time()
    expired = [sid for sid, s in _sessions.items() if now - s.last_active > SESSION_TTL]
    for sid in expired:
        del _sessions[sid]
    return len(_sessions)


def _build_agent_loop() -> AgentLoop:
    """Create a new AgentLoop from environment config."""
    # Clear proxy env vars that would route LLM API calls through a local proxy
    for key in ("all_proxy", "http_proxy", "https_proxy",
                "ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)

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
        thinking=os.environ.get("THINKING", "true").lower() not in ("0", "false", "no"),
        stream=True,
    )
    return AgentLoop(
        config,
        skills_dir=os.environ.get("SKILLS_DIR", "skills"),
        tasks_dir=os.environ.get("TASKS_DIR", ".hqagent/tasks"),
    )


def _get_session(sid: str) -> _Session:
    s = _sessions.get(sid)
    if s is None:
        raise HTTPException(status_code=404, detail=f"Session '{sid}' not found")
    s.last_active = time.time()
    return s


# ---------------------------------------------------------------------------
# Lifespan: periodic GC
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    async def gc_loop():
        while True:
            await asyncio.sleep(300)  # every 5 minutes
            remaining = _gc_sessions()
            if remaining:
                pass  # keep gc loop quiet in production

    task = asyncio.create_task(gc_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="hqagent API", version="0.1.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message to send to the agent.")
    session_id: str | None = Field(
        default=None,
        description="Existing session ID. If omitted, a new session is created.",
    )


class SessionInfo(BaseModel):
    id: str
    created_at: float
    last_active: float
    turn_count: int


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------


def _sse_event(event: str, data: dict | str) -> str:
    # Always JSON-encode so newlines inside string payloads don't break SSE framing
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_chat(loop: AgentLoop, user_message: str):
    """Run the agent loop and yield SSE events."""
    # Wire callbacks to push SSE events
    send_queue: asyncio.Queue = asyncio.Queue()

    def on_text(text: str) -> None:
        send_queue.put_nowait(("text", text))

    def on_thinking(chunk: str) -> None:
        send_queue.put_nowait(("thinking", chunk))

    def on_tool_call(name: str, inp: dict) -> None:
        send_queue.put_nowait(("tool_call", {"name": name, "input": inp}))

    def on_tool_result(name: str, result: str) -> None:
        send_queue.put_nowait(("tool_result", {"name": name, "result": result[:500]}))

    loop.on_text = on_text
    loop.on_thinking = on_thinking
    loop.on_tool_call = on_tool_call
    loop.on_tool_result = on_tool_result

    # Run the agent loop in a background task
    async def _run():
        try:
            final = await loop.run(user_message)
            await send_queue.put(("done", {"text": final, "usage": _usage_dict(loop.llm.usage)}))
        except Exception as exc:
            await send_queue.put(("error", str(exc)))

    task = asyncio.create_task(_run())

    # Drain the queue and yield SSE events
    done_received = False
    while not done_received:
        try:
            event, data = await asyncio.wait_for(send_queue.get(), timeout=0.1)
            if event == "done" or event == "error":
                done_received = True
            yield _sse_event(event, data)
        except asyncio.TimeoutError:
            # No message yet, keep looping — but yield a keepalive periodically
            if task.done():
                # Task finished without sending done (shouldn't happen)
                if task.exception():
                    yield _sse_event("error", str(task.exception()))
                else:
                    yield _sse_event("done", {})
                break

    await task


def _usage_dict(u: Usage) -> dict:
    return {
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_tokens": u.cache_read_tokens,
        "cache_write_tokens": u.cache_write_tokens,
        "total_tokens": u.total_tokens(),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/agent/health")
async def health():
    """Health check — returns provider/model info and session count."""
    return {
        "status": "ok",
        "model": os.environ.get("MODEL", "claude-sonnet-4-6"),
        "provider": os.environ.get("PROVIDER", "anthropic"),
        "sessions": len(_sessions),
    }


@app.get("/api/agent/sessions")
async def list_sessions():
    """List all active sessions."""
    _gc_sessions()
    result = []
    for sid, s in _sessions.items():
        result.append(
            SessionInfo(
                id=sid,
                created_at=s.created_at,
                last_active=s.last_active,
                turn_count=len(s.loop.context),
            )
        )
    return result


@app.post("/api/agent/sessions", status_code=201)
async def create_session():
    """Create a new empty agent session. Returns the session ID."""
    sid = uuid.uuid4().hex[:12]
    loop = _build_agent_loop()
    _sessions[sid] = _Session(loop)
    return {"session_id": sid}


@app.get("/api/agent/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session metadata."""
    s = _get_session(session_id)
    return SessionInfo(
        id=session_id,
        created_at=s.created_at,
        last_active=s.last_active,
        turn_count=len(s.loop.context),
    )


@app.delete("/api/agent/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and free its resources."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    del _sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.post("/api/agent/chat")
async def chat(req: ChatRequest):
    """Send a message to the agent and receive a streaming SSE response.

    Events emitted:
      - text:        A chunk of LLM output text.
      - tool_call:   A tool was invoked (name + input).
      - tool_result: A tool returned a result (name + truncated result).
      - done:        Agent finished (final text + token usage).
      - error:       An error occurred.
    """
    sid = req.session_id
    if sid is None or sid not in _sessions:
        sid = uuid.uuid4().hex[:12]
        _sessions[sid] = _Session(_build_agent_loop())

    s = _get_session(sid)

    return StreamingResponse(
        _stream_chat(s.loop, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": sid,
        },
    )


@app.get("/api/agent/skills")
async def list_skills():
    """List available skills."""
    skills_dir = os.environ.get("SKILLS_DIR", "skills")
    loader = SkillLoader(skills_dir)
    return {
        "skills": [
            {"name": s.name, "description": s.description}
            for s in loader._skills.values()
        ]
    }


@app.get("/api/agent/tools")
def list_tools():
    """List all registered tools with their schemas."""
    loop = _build_agent_loop()
    return {"tools": loop.registry.schemas()}
