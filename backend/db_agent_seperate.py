import asyncio

from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain.chat_models import init_chat_model

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langsmith import Client

from db import engine

SQL_PROMPT = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct postgres query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 10 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables.
"""

def init_agent(engine):
    sql_db = SQLDatabase(engine)
    llm = init_chat_model("gemini-2.5-flash" , model_provider="google_genai")
    prompt = SQL_PROMPT.strip()
    toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)

    agent = create_agent(llm, toolkit.get_tools(), system_prompt=prompt)
    return agent

async def stream_agent(agent, question: str) -> None:
    """Stream the agent's run: tool calls, tool results and answer tokens as they arrive."""
    # stream_mode="messages" yields (message_chunk, metadata) for every LLM token
    # and every tool result, instead of waiting for whole steps to finish.
    async for chunk, metadata in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="messages",
    ):
        if isinstance(chunk, ToolMessage):
            print(f"\n[tool result: {chunk.name}] {chunk.text[:200]}")
        elif isinstance(chunk, AIMessageChunk):
            for call in chunk.tool_calls:
                print(f"\n[tool call: {call['name']}] {call['args']}")
            if chunk.text:
                print(chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    agent = init_agent(engine)
    asyncio.run(stream_agent(agent, "who is the topper"))