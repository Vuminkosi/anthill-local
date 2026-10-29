"""Real MCP stdio protocol test; readiness calls the local API without inference."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
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
        environment = os.environ.copy()
        environment.update(
            ANTHILL_URL="http://127.0.0.1:8090",
            # Readiness checks only model availability and never posts a story.
            OLLAMA_URL="http://127.0.0.1:1",
            PYTHONPATH=str(APP_DIR),
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[str(APP_DIR / "mcp_server.py")],
            env=environment,
        )

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

        asyncio.run(exercise_protocol())


if __name__ == "__main__":
    unittest.main()
