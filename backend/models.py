from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    # Conversation id: questions with the same thread_id share memory.
    # Omitted → a fresh one-off conversation.
    thread_id: str | None = None


class ConnectRequest(BaseModel):
    host: str
    port: int = 5432
    username: str
    password: str
    database: str
    ssl: str = "require"
    type: str = "PostgreSQL"
