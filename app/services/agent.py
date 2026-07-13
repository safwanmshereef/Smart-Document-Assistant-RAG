import os
from typing import List, Dict, Any

# Import base LangChain components (with classic agents fallback)
try:
    from langchain.agents import create_tool_calling_agent, AgentExecutor
except ImportError:
    from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import internal services
from sqlalchemy.orm import Session
from app.services.llm_factory import get_llm
from app.services.tools import search_documents, calculator, get_current_date_time, web_search, summarize_document_topic
from app.database.config import SessionLocal
from app.database.models import ChatMessage as DbChatMessage

def load_chat_history(session_id: str, db: Session = None) -> List[BaseMessage]:
    """
    Retrieves all past messages for a session from the SQLite database
    and formats them as LangChain human/AI messages.

    Args:
        session_id: The session ID to fetch history for.
        db: Optional SQLAlchemy Session.

    Returns:
        A list of LangChain BaseMessage objects representing the session history.
    """
    db_session = db if db is not None else SessionLocal()
    try:
        db_messages = (
            db_session.query(DbChatMessage)
            .filter(DbChatMessage.session_id == session_id)
            .order_by(DbChatMessage.timestamp.asc())
            .all()
        )
        history = []
        for msg in db_messages:
            if msg.role == "user":
                history.append(HumanMessage(content=msg.content))
            elif msg.role == "agent":
                history.append(AIMessage(content=msg.content))
        return history
    finally:
        if db is None:
            db_session.close()

def save_chat_message(session_id: str, role: str, content: str, db: Session = None) -> None:
    """
    Persists a single chat message into the SQLite database.

    Args:
        session_id: The active session ID.
        role: Message role (either 'user' or 'agent').
        content: The message text content.
        db: Optional SQLAlchemy Session.
    """
    db_session = db if db is not None else SessionLocal()
    try:
        db_message = DbChatMessage(session_id=session_id, role=role, content=content)
        db_session.add(db_message)
        db_session.commit()
    finally:
        if db is None:
            db_session.close()


class DocumentAssistantAgent:
    """
    Agent class representing the Smart Document Assistant.
    Orchestrates the LLM, binds tools, and wraps execution inside an AgentExecutor.
    """
    def __init__(self, provider: str, model_name: str = None):
        self.llm = get_llm(provider, model_name)
        self.tools = [
            search_documents,
            calculator,
            get_current_date_time,
            web_search,
            summarize_document_topic
        ]

        # Injects the strict anti-hallucination and citation system instructions
        system_prompt = (
            "You are a Smart Document Assistant.\n\n"
            "When answering questions based on documents, you MUST append the exact Source and Page citations "
            "provided by the search tool.\n\n"
            "If the user asks a question and the information is NOT contained in the retrieved documents, "
            "you MUST strictly reply with 'I don't know' or 'The provided documents do not contain this information.' "
            "Do not hallucinate, guess, or use outside knowledge to answer document-specific queries."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Bind tools to the LLM and create tool calling agent
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        # Wrap in executor to track intermediate thoughts and tool executions
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True
        )

    def run(self, user_message: str, chat_history: List[BaseMessage]) -> Dict[str, Any]:
        """
        Executes the agent loop.

        Args:
            user_message: New prompt from the user.
            chat_history: Chat history message logs.

        Returns:
            The raw execution output dictionary.
        """
        return self.executor.invoke({
            "input": user_message,
            "chat_history": chat_history
        })


def chat_with_agent(
    session_id: str, 
    user_message: str, 
    provider: str, 
    model_name: str = None,
    db: Session = None
) -> Dict[str, Any]:
    """
    Main stateful entry point for interacting with the Smart Document Assistant.
    Retrieves history from SQLite, runs the agentic loop, writes logs back to SQLite,
    and returns final text + reasoning execution traces.

    Args:
        session_id: Session identifier.
        user_message: Input text query.
        provider: LLM Provider ('google' or 'ollama').
        model_name: Optional custom model name.
        db: Optional SQLAlchemy Session.

    Returns:
        A dictionary containing:
        - "output": The final textual answer.
        - "reasoning_trace": List of intermediate tool calls with parameters and outputs.
    """
    # 1. Fetch historical logs (excludes the current input message)
    chat_history = load_chat_history(session_id, db=db)

    # 2. Write the user's incoming query to the database
    save_chat_message(session_id, "user", user_message, db=db)

    # 3. Instantiate the agent core
    agent = DocumentAssistantAgent(provider=provider, model_name=model_name)

    # 4. Invoke the agent execution
    result = agent.run(user_message, chat_history)
    output = result.get("output", "")

    # Safely convert output to a clean string if it's returned as a list/dict by some models
    if isinstance(output, list):
        text_parts = []
        for part in output:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        output = "\n".join(text_parts)
    elif not isinstance(output, str):
        output = str(output)

    # 5. Write the agent's response to the database
    save_chat_message(session_id, "agent", output, db=db)

    # 6. Parse reasoning trace
    reasoning_trace = []
    for action, observation in result.get("intermediate_steps", []):
        reasoning_trace.append({
            "tool": action.tool,
            "tool_input": action.tool_input,
            "output": str(observation)
        })

    return {
        "output": output,
        "reasoning_trace": reasoning_trace
    }
