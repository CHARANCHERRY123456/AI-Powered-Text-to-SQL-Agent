import asyncio

from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain.chat_models import init_chat_model

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from config import LLM_MODEL
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
    llm = init_chat_model(LLM_MODEL, model_provider="google_genai")
    toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)

    return create_agent(
        llm,
        toolkit.get_tools(),
        system_prompt=SQL_PROMPT.strip(),
        checkpointer=InMemorySaver(),  # in-process memory, lost when the script exits
    )


async def stream_agent(agent, question: str, thread_id: str) -> None:
    """Stream the agent's run: tool calls, tool results and answer tokens as they arrive."""
    config = {"configurable": {"thread_id": thread_id}}
    # stream_mode="messages" yields (message_chunk, metadata) for every LLM token
    # and every tool result, instead of waiting for whole steps to finish.
    async for chunk, metadata in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(chunk, ToolMessage):
            print(f"\n[tool result: {chunk.name}] {chunk.text[:200]}")
        # Only the agent's own LLM runs in the "model" node. The query-checker tool
        # also calls the LLM (inside the "tools" node); skip its tokens.
        elif isinstance(chunk, AIMessageChunk) and metadata.get("langgraph_node") == "model":
            for call in chunk.tool_calls:
                print(f"\n[tool call: {call['name']}] {call['args']}")
            if chunk.text:
                print(chunk.text, end="", flush=True)
    print()


async def chat(agent, thread_id: str = "thread_1") -> None:
    while True:
        try:
            # input() blocks, so run it in a thread to keep the event loop free.
            question = (await asyncio.to_thread(input, "\nYour question: ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        try:
            await stream_agent(agent, question, thread_id)
        except Exception as exc:  # e.g. LLM rate limit or DB outage — keep the chat alive
            print(f"\n[error] {exc}")


if __name__ == "__main__":
    agent = init_agent(engine)
    try:
        asyncio.run(chat(agent))
    except KeyboardInterrupt:  # Ctrl+C while the agent is running
        pass
    print("Bye!")