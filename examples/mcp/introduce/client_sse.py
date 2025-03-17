"""
https://github.com/modelcontextprotocol/quickstart-resources/blob/main/mcp-client-python/client.py
基于mcp官方例子改造，使用openai的client，与官方例子使用的anthrotic的client有些地方不一样，下面标记“注意点”来区分
"""
import ast
import asyncio
import os
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI

load_dotenv()  # load environment variables from .env

BASE_URL = os.environ.get("BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL = os.environ.get("MODEL")


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

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

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [{"role": "user", "content": query}]

        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,  # 注意点：anthrotic使用“input_schema”
                },
            }
            for tool in response.tools
        ]

        # Initial LLM API call
        # 注意点：调用入口不一样
        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
        )

        # Process response and handle tool calls
        final_text = []

        # 注意点：后面的结果解析都不一样
        message = response.choices[0].message

        # Add assistant's message to conversation
        messages.append(
            {
                "role": "assistant",
                "content": message.content if message.content else None,
                "tool_calls": message.tool_calls if hasattr(message, "tool_calls") else None,
            }
        )

        # If no tool calls, we're done
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            if message.content:
                final_text.append(message.content)
            return message.content or ""

        # Handle tool calls
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments

            # Convert tool_args from string to dictionary if necessary
            try:
                tool_args = ast.literal_eval(tool_args_str)
            except (ValueError, SyntaxError):
                # 如果解析失败，尝试使用更简单的方法
                import json

                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    print(f"Error parsing tool arguments: {tool_args_str}")
                    tool_args = {}

            # Ensure tool_args is a dictionary
            if not isinstance(tool_args, dict):
                tool_args = {}

            # Execute tool call
            print(f"Calling tool {tool_name} with args: {tool_args}")
            try:
                result = await self.session.call_tool(tool_name, tool_args)
                tool_result = str(result.content)
                final_text.append(f"[Calling tool {tool_name}]")
            except Exception as e:
                print(f"Error calling tool {tool_name}: {e}")
                tool_result = f"Error: {str(e)}"
                final_text.append(f"[Error calling tool {tool_name}: {str(e)}]")

            # Add tool result to messages
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": tool_result, "name": tool_call.function.name}
            )

        # Get next response from Claude
        response = self.client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
        )

        message = response.choices[0].message
        if message.content:
            final_text.append(message.content)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def chat_once(self):
        query = "How old is Bob?"

        try:
            response = await self.process_query(query)
            print("\n" + response)
        except Exception as e:
            print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    server_url = "http://localhost:8000/sse"

    client = MCPClient()
    try:
        await client.connect_to_server(server_url)
        await client.chat_once()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
