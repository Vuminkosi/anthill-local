"""Real MCP stdio protocol test; readiness calls the local API without inference."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"

try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError as exc:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""


@unittest.skipIf(ClientSession is None, f"MCP dependencies unavailable: {_IMPORT_ERROR}")
class McpStdioTests(unittest.TestCase):
    def test_stdio_initialize_list_tools_and_call_local_readiness(self) -> None:
        requested_paths = []

        class ReadinessHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                requested_paths.append(self.path)
                if self.path != "/api/readiness":
                    self.send_error(404)
                    return
                body = json.dumps(
                    {
                        "status": "models-needed",
                        "models": {
                            "image": {"installed": False},
                            "narration": {"installed": False},
                            "storyboard": {"installed": False},
                            "animation": {"installed": False},
                        },
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args) -> None:
                pass

        readiness_server = ThreadingHTTPServer(("127.0.0.1", 0), ReadinessHandler)
        readiness_thread = threading.Thread(target=readiness_server.serve_forever, daemon=True)
        readiness_thread.start()

        async def exercise_protocol() -> None:
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    initialized = await session.initialize()
                    self.assertEqual(initialized.serverInfo.name, "AntHill Local Video Factory")
                    tools = await session.list_tools()
                    names = {tool.name for tool in tools.tools}
                    self.assertEqual(
                        names,
                        {
                            "generate_video",
                            "get_video_job",
                            "list_video_jobs",
                            "retry_video_job",
                            "check_readiness",
                        },
                    )
                    readiness = await session.call_tool("check_readiness", {})
                    self.assertFalse(readiness.isError, repr(readiness.content))
                    result = readiness.structuredContent
                    if result is None:
                        result = json.loads(readiness.content[0].text)
                    self.assertIsInstance(result, dict)
                    self.assertEqual(result["readiness"]["status"], "models-needed")
                    models = result["readiness"]["models"]
                    self.assertFalse(models["image"]["installed"])
                    self.assertFalse(models["narration"]["installed"])
                    self.assertFalse(models["storyboard"]["installed"])
                    self.assertFalse(models["animation"]["installed"])

        try:
            environment = os.environ.copy()
            environment.update(
                ANTHILL_URL=f"http://127.0.0.1:{readiness_server.server_address[1]}",
                # The MCP child calls only the test-owned readiness stub.
                OLLAMA_URL="http://127.0.0.1:1",
                PYTHONPATH=str(APP_DIR),
            )
            parameters = StdioServerParameters(
                command=sys.executable,
                args=[str(APP_DIR / "mcp_server.py")],
                env=environment,
            )
            asyncio.run(exercise_protocol())
        finally:
            readiness_server.shutdown()
            readiness_server.server_close()
            readiness_thread.join(timeout=2)

        self.assertEqual(requested_paths, ["/api/readiness"])


if __name__ == "__main__":
    unittest.main()
