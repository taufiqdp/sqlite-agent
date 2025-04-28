from contextlib import AsyncExitStack

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from agent.sub_agents.python_agent.agent import get_python_agent
from agent.sub_agents.sqlite_agent.agent import get_sqlite_agent


async def create_main_agent():
    exit_stack = AsyncExitStack()
    await exit_stack.__aenter__()

    sqlite_agent, sqlite_stack = await get_sqlite_agent()
    await exit_stack.enter_async_context(sqlite_stack)

    python_agent, python_stack = await get_python_agent()
    await exit_stack.enter_async_context(python_stack)

    root_agent = Agent(
        model=LiteLlm("azure/gpt-4o-mini"),
        name="main_agent",
        sub_agents=[sqlite_agent, python_agent],
        instruction="""You are a helpful AI assistant. Your goal is to answer user requests accurately and efficiently.
        You have access to specialized sub-agents (tools) to help you with specific tasks.

        <Sub-Agent Capabilities>
        - `sqlite_agent`: Use this agent when you need to interact with the SQLite database. This includes querying data, checking table schemas, or retrieving specific information stored in the database.
        - `python_agent`: Use this agent when you need to execute Python code. This is useful for calculations, data manipulation, running scripts, or any task requiring programming logic.
        </Sub-Agent Capabilities>

        <Workflow>
        1. Analyze the user's request.
        2. Determine if you can answer the request directly based on your general knowledge.
        3. If the request requires accessing database information, formulate a plan to use the `sqlite_agent` to retrieve the necessary data.
        4. If the request requires executing Python code, formulate a plan to use the `python_agent`.
        5. If the request requires both database access and code execution (e.g., fetch data then process it), plan the steps accordingly:
            a. Call `sqlite_agent` to get the data.
            b. Call `python_agent` with the retrieved data to perform the calculations or manipulations.
        6. Synthesize the information gathered from sub-agents (if any) and your own knowledge to provide a comprehensive answer to the user.
        </Workflow>

        <Key Constraints>
        - Use the `sqlite_agent` ONLY for database-related tasks.
        - Use the `python_agent` ONLY for Python code execution tasks.
        - If a sub-agent is needed, clearly state your plan before calling it.
        - Combine information effectively if multiple steps or agents are involved.
        - Answer the user's query directly if no specialized tools are needed.
        </Key Constraints>
        """,
    )

    return root_agent, exit_stack


root_agent = create_main_agent()
