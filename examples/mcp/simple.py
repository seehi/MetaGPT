import asyncio
from contextlib import AsyncExitStack
from functools import partial
from typing import Optional

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import Tool as MCPTool

from metagpt.roles.di.role_zero import RoleZero
from metagpt.schema import Message
from metagpt.tools.tool_data_type import Tool
from metagpt.tools.tool_registry import TOOL_REGISTRY

SERVER_URL = "http://localhost:8000/sse"


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
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
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])
        return tools

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


class Master(RoleZero):
    name: str = "Master"
    tools: list[str] = ["get_user_age"]

    mcp_client: Optional[MCPClient] = None
    mcp_tools: list[MCPTool] = []

    async def _update_mcp_tool_execution(self):
        async def mcp_tool_adapter(tool_name, **kwargs):
            return await self.mcp_client.session.call_tool(tool_name, arguments=kwargs)

        for mcp_tool in self.mcp_tools:
            self.tool_execution_map.update({mcp_tool.name: partial(mcp_tool_adapter, mcp_tool.name)})

    async def _quick_think(self) -> tuple[Message, str]:
        return None, ""


async def register_mcp_tools() -> tuple[MCPClient, list[MCPTool]]:
    def register_mcp_tool(tool: MCPTool):
        schema = {"description": tool.description, "parameters": tool.inputSchema}

        tool = Tool(name=tool.name, schemas=schema, path="")
        TOOL_REGISTRY.tools[tool.name] = tool

    mcp_client = MCPClient()
    await mcp_client.connect_to_server(SERVER_URL)
    tools = await mcp_client.list_tools()
    for tool in tools:
        register_mcp_tool(tool)

    return mcp_client, tools


async def main():
    mcp_client, mcp_tools = await register_mcp_tools()

    role = Master(mcp_client=mcp_client, mcp_tools=mcp_tools)
    await role._update_mcp_tool_execution()
    try:
        await role.run(Message(content="What is the age of the user named Bob?", send_to={role.name}))
    finally:
        if role.mcp_client:
            await role.mcp_client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
