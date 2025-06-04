import asyncio
import subprocess
import time
from contextlib import AsyncExitStack
from functools import partial
from typing import Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import Tool as MCPTool
from pydantic import model_validator

from metagpt.const import EXAMPLE_PATH
from metagpt.roles.di.role_zero import RoleZero
from metagpt.schema import Message
from metagpt.tools.tool_data_type import Tool
from metagpt.tools.tool_registry import TOOL_REGISTRY


class MCPClient:
    def __init__(self):
        """Initialize session and client objects"""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_url: str):
        """Connect to an MCP server

        Args:
            server_url: The url of server.
        """
        stdio_transport = await self.exit_stack.enter_async_context(sse_client(server_url))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        try:
            await self.session.initialize()
        except Exception as e:
            print(f"Error initializing session: {e}")

        print(f"Connect to server:{server_url} success!")

    async def list_tools(self) -> list[MCPTool]:
        """List available tools"""
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        return tools

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


class Master(RoleZero):
    name: str = "Master"

    mcp_client: Optional[MCPClient] = None
    mcp_tools: list[MCPTool] = []

    @model_validator(mode="after")
    def set_plan_and_tool(self) -> "Master":
        if self.mcp_tools:
            self.tools = [tool.name for tool in self.mcp_tools]

        return super().set_plan_and_tool()

    async def __aenter__(self):
        await self._update_mcp_tool_execution()
        return self

    async def __aexit__(self, *args, **kwargs):
        if self.mcp_client:
            await self.mcp_client.cleanup()

    async def _update_mcp_tool_execution(self):
        """Setup mcp tool execution, when call mcp tool, use MCP Client to call the MCP Server"""

        async def mcp_tool_adapter(tool_name, **kwargs):
            return await self.mcp_client.session.call_tool(tool_name, arguments=kwargs)

        for mcp_tool in self.mcp_tools:
            self.tool_execution_map.update({mcp_tool.name: partial(mcp_tool_adapter, mcp_tool.name)})

    async def _quick_think(self) -> tuple[Message, str]:
        return None, ""


async def register_mcp_tools(server_url: str) -> tuple[MCPClient, list[MCPTool]]:
    """Connect to MCP server, list tools and register them to TOOL_REGISTRY"""

    def register_mcp_tool(tool: MCPTool):
        schema = {"description": tool.description, "parameters": tool.inputSchema}

        tool = Tool(name=tool.name, schemas=schema, path="")
        TOOL_REGISTRY.tools[tool.name] = tool

    mcp_client = MCPClient()
    await mcp_client.connect_to_server(server_url)
    tools = await mcp_client.list_tools()
    for tool in tools:
        register_mcp_tool(tool)

    return mcp_client, tools


def start_mcp_server() -> tuple[subprocess.Popen, str]:
    """Create a process to run the MCP Server"""
    try:
        server_file = EXAMPLE_PATH / "mcp" / "introduce" / "server_sse.py"
        server_url = "http://localhost:8000/sse"

        print(f"Starting SSE server at {server_url} ...")

        process = subprocess.Popen(["python", server_file])
        # Give it 5 seconds to start
        time.sleep(5)

        print("SSE server started. Running example...\n\n")
        return process, server_url
    except Exception as e:
        print(f"Error starting SSE server: {e}")
        exit(1)


async def main():
    # 1. Start the MCP Server
    process, server_url = start_mcp_server()

    try:
        # 2. There is only one MCP Tool named `get_user_age` on the MCP Server
        mcp_client, mcp_tools = await register_mcp_tools(server_url)

        # 3. Ask Bob's age; the command should looks like: ```json [{"command_name": "get_user_age", "args": {"user_name": "Bob"}}]```
        async with Master(mcp_client=mcp_client, mcp_tools=mcp_tools) as role:
            msg = Message(content="What is the age of the user named Bob?", send_to={role.name})
            await role.run(msg)
    finally:
        process.kill()


if __name__ == "__main__":
    asyncio.run(main())
