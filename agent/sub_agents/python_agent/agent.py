from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import (MCPToolset,
                                                   StdioServerParameters)

load_dotenv()

INSTRUCTION = """<Role>
You are an AI assistant specialized in executing Python code.
</Role>

<Capabilities>
You have access to a tool that allows you to execute Python code snippets.
</Capabilities>

<Workflow>
1. Receive a request that requires Python code execution (e.g., calculations, data manipulation, scripting).
2. Understand the goal of the required Python code.
3. If the user provides the code, review it for safety and relevance.
4. If you need to generate the code, formulate the necessary Python script.
5. Use the tool to execute the code snippet.
6. Capture the output (stdout, stderr) and any results from the execution.
7. Present the execution results, including any output or errors, clearly to the user or the requesting agent.
</Workflow>

<Key Constraints>
- Only use the tool for executing Python code.
- Ensure the code to be executed is safe and directly relevant to the task.
- Do not execute code that performs file system operations, network requests, or other potentially harmful actions unless explicitly part of the agreed-upon task and capabilities.
- Clearly report any errors encountered during code execution.
</Key Constraints>
"""


async def get_tools():
    """
    Initializes and establishes connection to the MCP Python server via Deno.

    Returns:
        tuple: A tuple containing:
            - tools (MCPToolset): Collection of MCP tools/commands for Python execution.
            - exit_stack (AsyncExitStack): Context manager for cleanup.
    """
    server_params = StdioServerParameters(
        command="deno",
        args=[
            "run",
            "-N",
            "-R=node_modules",
            "-W=node_modules",
            "--node-modules-dir=auto",
            "jsr:@pydantic/mcp-run-python",
            "stdio",
        ],
    )

    tools, exit_stack = await MCPToolset.from_server(connection_params=server_params)

    return tools, exit_stack


async def get_python_agent():
    """
    Initializes and returns the Python execution agent and its associated exit stack.

    Returns:
        tuple: A tuple containing:
            - root_agent (Agent): The configured Python agent.
            - exit_stack (AsyncExitStack): Context manager for cleanup.
    """
    tools, exit_stack = await get_tools()

    print(f"Successfully load {len(tools)} tools")

    root_agent = Agent(
        name="python_agent",
        model=LiteLlm("azure/gpt-4o-mini"),
        tools=tools,
        instruction=INSTRUCTION,
    )

    return root_agent, exit_stack
