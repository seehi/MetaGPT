from mcp.server.fastmcp import FastMCP

mcp = FastMCP("users")

DATA = {"Emma": 10, "Bob": 12}


@mcp.tool()
async def get_user_age(user_name: str) -> int:
    """Get user age for the user.

    Args:
        user_name: The name of user
    """

    return DATA.get(user_name, 0)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="sse")
