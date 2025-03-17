from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("users")

ages = {"Emma": 10, "Bob": 12}


@mcp.tool()
async def get_user_age(user_name: str) -> int:
    """Get user age for the user.

    Args:
        user_name: The name of user
    """

    return ages.get(user_name, 0)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="sse")
