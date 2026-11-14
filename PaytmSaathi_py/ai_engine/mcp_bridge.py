"""MCP client bridge: a synchronous wrapper around one long-lived MCP server session.

Streamlit and FastAPI code is synchronous, while the MCP client is async. The bridge runs the
session on its own thread/event loop, starts the Saathi MCP server (``mcp_server/server.py``) as a
subprocess once, and exposes ``list_tools()`` and ``call_tool()``.

Security: the model never supplies ``user_id`` / ``policy_id``. :meth:`call_tool` removes any such
arguments and fills in the trusted ones passed by the app, and :meth:`openai_tools` hides them from
the tool schemas the model sees.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Optional

from .assistant_tools import CONTEXT_PARAMS
from .utils import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
START_TIMEOUT_S = 90  # first start imports the whole engine
CALL_TIMEOUT_S = 90


class McpBridge:
    def __init__(self, command: Optional[str] = None, args: Optional[list[str]] = None):
        self.command = command or sys.executable
        self.args = args or ["-m", "mcp_server.server"]
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session = None
        self._ready = threading.Event()
        self._stop: Optional[asyncio.Event] = None
        self._error: Optional[str] = None
        self._tools_cache: Optional[list[dict]] = None
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive() and self._ready.is_set():
                return
            self._ready.clear()
            self._error = None
            self._thread = threading.Thread(target=self._run_loop, name="mcp-bridge", daemon=True)
            self._thread.start()
        if not self._ready.wait(START_TIMEOUT_S) or self._error:
            raise RuntimeError(self._error or "The MCP server did not start in time.")

    def _run_loop(self) -> None:
        # Tornado (used by Streamlit) switches Windows to the Selector loop, which cannot run subprocesses.
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            logger.warning("MCP bridge stopped: %s", self._error)
        finally:
            self._session = None
            self._ready.set()  # unblock waiters even on failure
            loop.close()

    async def _serve(self) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(command=self.command, args=self.args, cwd=str(PROJECT_ROOT),
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
        self._stop = asyncio.Event()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                self._ready.set()
                await self._stop.wait()

    def close(self) -> None:
        if self._loop and self._stop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and self._session is not None)

    # -- tools ---------------------------------------------------------------
    def _run(self, coro, timeout: float):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except FutureTimeout:
            future.cancel()
            raise TimeoutError("The tool took too long.")

    def _ensure(self) -> None:
        """Start (or restart after a crash) the server session before building any coroutine on it."""
        if not self.alive:
            self._tools_cache = None
            self.start()

    def list_tools(self) -> list[Any]:
        self._ensure()
        return self._run(self._session.list_tools(), CALL_TIMEOUT_S).tools

    def openai_tools(self) -> list[dict]:
        """Tool schemas in OpenAI function format, without the trusted context parameters."""
        if self._tools_cache is None:
            tools = []
            for t in self.list_tools():
                schema = json.loads(json.dumps(t.inputSchema))
                props = schema.get("properties", {})
                for p in CONTEXT_PARAMS:
                    props.pop(p, None)
                schema["required"] = [r for r in schema.get("required", []) if r not in CONTEXT_PARAMS]
                tools.append({"type": "function", "function": {
                    "name": t.name, "description": (t.description or "")[:900], "parameters": schema}})
            self._tools_cache = tools
        return self._tools_cache

    def call_tool(self, name: str, arguments: dict, context: Optional[dict] = None) -> Any:
        """Call a tool. ``context`` holds trusted user_id/policy_id that always override the model's args."""
        args = {k: v for k, v in (arguments or {}).items() if k not in CONTEXT_PARAMS}
        for key, value in (context or {}).items():
            if key in CONTEXT_PARAMS and value:
                args[key] = value
        self._ensure()
        result = self._run(self._session.call_tool(name, args), CALL_TIMEOUT_S)
        text = "\n".join(getattr(c, "text", "") for c in result.content)
        if getattr(result, "isError", False):
            return {"error": text[:300] or "tool error"}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"result": text}


_BRIDGE: Optional[McpBridge] = None
_BRIDGE_LOCK = threading.Lock()


def get_bridge() -> McpBridge:
    """Process-wide bridge (started lazily on first use)."""
    global _BRIDGE
    with _BRIDGE_LOCK:
        if _BRIDGE is None:
            _BRIDGE = McpBridge()
        return _BRIDGE
