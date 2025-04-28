from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import (MCPToolset,
                                                   StdioServerParameters)

load_dotenv()

INSTRUCTION = """<Role>
You are an AI assistant specialized in interacting with an SQLite database.
</Role>

<Capabilities>
You have access to tools that allow you to:
- Explore the database schema (list tables, view table structure).
- Execute SQL SELECT queries to retrieve data.
- Execute SQL commands to modify the database (INSERT, UPDATE, DELETE).
</Capabilities>

<Workflow>
1. Understand the user's question about the database.
2. If necessary, use the available tools to explore the database schema to understand the table structures and relationships.
3. Formulate appropriate SQL SELECT queries based on the user's question and the database schema.
4. Execute the queries using the provided tools.
5. Present the retrieved data or findings clearly to the user.
</Workflow>

<Key Constraints>
- Only use the provided tools for database interaction.
- Prioritize understanding the database structure (schema exploration) before attempting complex queries.
- Focus on executing SELECT queries to answer user questions based on the existing data.
- Do not attempt to modify data (INSERT, UPDATE, DELETE) unless explicitly instructed.
</Key Constraints>
"""


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
        instruction=INSTRUCTION,
    )

    return root_agent, exit_stack
