from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import (MCPToolset,
                                                   StdioServerParameters)

load_dotenv()


async def get_tools():
    """
    Initializes and establishes connection to MCP SQLite server to obtain tools.

    Returns:
        tuple: A tuple containing:
            - tools (MCPToolset): Collection of MCP tools/commands
            - exit_stack (AsyncExitStack): Context manager for cleanup
    """
    print("Initializing tools...")
    server_params = StdioServerParameters(
        command="uvx", args=["mcp-server-sqlite", "--db-path", "mydb.sqlite"]
    )

    print("Connecting to SQLite server...")
    tools, exit_stack = await MCPToolset.from_server(connection_params=server_params)
    print("Tools initialized successfully")

    return tools, exit_stack


async def get_sqlite_agent():
    """
    Initializes the MCP tools and creates an Agent instance configured for SQLite interaction.

    Returns:
        tuple: A tuple containing:
            - root_agent (Agent): The configured agent instance.
            - exit_stack (AsyncExitStack): Context manager for cleaning up tool connections.
    """
    tools, exit_stack = await get_tools()

    print(f"Successfully load {len(tools)} tools")

    root_agent = Agent(
        name="sqlite_agent",
        model=LiteLlm("azure/gpt-4o-mini"),
        tools=tools,
        instruction="""You are an AI assistant designed to interact with an SQLite database.
        Use the available tools to explore the database schema and execute SQL SELECT queries to answer user questions based on the data.
        Prioritize understanding the database structure before attempting complex queries.
        Present the results clearly.""",
    )

    return root_agent, exit_stack


