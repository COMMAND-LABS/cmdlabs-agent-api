"""
Agent completion endpoint — non-streaming.

Runs the agent to completion and returns the full output as a single JSON
response.  Uses the same agent setup logic as stream.py but calls ainvoke
instead of astream_events so no SSE is involved.
"""
from typing import Optional
import time
import uuid
from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError
from src.deps import db_dependency, auth_dependency
from src.db.models import Agent, ChatSession, ChatMessage, Credential
from src.db.service_name import ServiceName
from src.routers.agents.access import load_agent_with_access_check
from src.routers.credentials.encryption import get_credential_value
from src.core.schemas.ChatSessionPrompt import ChatSessionPrompt
from slowapi import Limiter
from slowapi.util import get_remote_address
from dotenv import load_dotenv

from src.utils.langsmith import get_langsmith_callbacks
from src.utils.template_variables import resolve_template_variables, build_variable_context
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent, create_tool_calling_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.tools import create_tools_from_agent_config, CredentialError
from src.utils.pdf_to_images import build_pdf_message
from src.routers.agents.helpers import (
    build_message_history,
    store_user_message,
    store_ai_message,
    extract_auth_token,
    format_tool_call,
    get_model_config,
    create_llm,
    get_required_credential_type,
)

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

load_dotenv()

callbacks = get_langsmith_callbacks("dynamic-agent-completion")


def _is_transient_ssl_db_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "ssl connection has been closed unexpectedly" in text
        or "server closed the connection unexpectedly" in text
        or "connection reset by peer" in text
        or "could not receive data from server" in text
    )


def _db_retry_once(db, operation_name: str, fn):
    try:
        return fn()
    except OperationalError as e:
        if not _is_transient_ssl_db_error(e):
            raise
        print(f"[AGENT COMPLETION] Transient DB SSL error during {operation_name}; retrying once...")
        try:
            db.rollback()
        except Exception:
            pass
        db.close()
        time.sleep(0.5)
        return fn()


@router.post("/{agent_id}/completion")
@limiter.limit("60/minute")
async def agent_completion(
    agent_id: int,
    request_body: ChatSessionPrompt,
    db: db_dependency,
    auth: auth_dependency,
    request: Request,
):
    """
    Run an agent to completion and return the full output as JSON.

    Returns:
        {
            "output": "<full AI response>",
            "tool_calls": [...],   // empty list if no tools were used
            "session_id": "<uuid>",
            "agent_id": <int>
        }
    """
    print(f"[AGENT COMPLETION] Received non-streaming request for agent_id: {agent_id}")

    try:
        account_id = int(auth['id']) if isinstance(auth['id'], str) else auth['id']

        agent = _db_retry_once(
            db, "load agent",
            lambda: load_agent_with_access_check(db, account_id, agent_id),
        )

        if not agent:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found or access denied.")

        if not agent.config:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Agent configuration is missing.")

        config_data = agent.config.get('data', {})
        system_prompt_raw = config_data.get('systemPrompt', 'You are a helpful assistant.')
        var_context = build_variable_context(agent_name=agent.name)
        system_prompt_resolved = resolve_template_variables(system_prompt_raw, var_context)
        system_prompt = system_prompt_resolved.replace("{", "{{").replace("}", "}}")

        model_config = get_model_config(agent.config)
        provider = model_config['provider']
        model_name = model_config['model']
        print(f"[AGENT COMPLETION] Using model: {provider}/{model_name}")

        required_credential_type = get_required_credential_type(provider)
        credentials = {}

        if required_credential_type:
            credential = _db_retry_once(
                db, "load provider credential",
                lambda: db.query(Credential).filter(
                    Credential.account_id == account_id,
                    Credential.credential_type == required_credential_type
                ).first(),
            )

            if not credential:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Please add your {provider.title()} API key in account settings to use {model_name}."
                )

            try:
                api_key = get_credential_value(credential, "api_key")
                credentials[provider] = api_key
            except Exception as e:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve API key: {e}")

        try:
            llm, llm_provider = create_llm(
                model_config=model_config,
                credentials=credentials,
                streaming=False,
                temperature=0,
            )
            print(f"[AGENT COMPLETION] Initialized {llm_provider} LLM: {model_name}")
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"LLM initialization failed: {e}")

        try:
            session_uuid = uuid.UUID(request_body.sessionId)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="sessionId must be a valid UUID.")

        session = _db_retry_once(
            db, "load chat session",
            lambda: db.query(ChatSession).filter(
                ChatSession.session_id == session_uuid,
                ChatSession.account_id == account_id
            ).first(),
        )

        if not session:
            print(f"[AGENT COMPLETION] Session {request_body.sessionId} not found, creating automatically...")
            try:
                session = ChatSession(
                    session_id=session_uuid,
                    agent_id=agent_id,
                    account_id=account_id,
                    title=f"Chat with Agent {agent_id}"
                )
                db.add(session)
                db.commit()
                db.refresh(session)
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not create chat session: {e}")

        db_messages = _db_retry_once(
            db, "load chat messages",
            lambda: db.query(ChatMessage).filter(
                ChatMessage.chat_session_id == session.id
            ).order_by(ChatMessage.created_at.asc()).all(),
        )

        message_history = build_message_history(db_messages)
        auth_token = extract_auth_token(request, auth)

        try:
            tools = await create_tools_from_agent_config(
                agent_config=agent.config,
                account_id=account_id,
                db=db,
                auth_token=auth_token,
                request=request,
                chat_session_id=session_uuid,
                agent_owner_account_id=agent.account_id,
            )
        except CredentialError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Tool configuration error: {e}")
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid tool configuration: {e}")

        # Build prompt template
        if tools:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True,
            output_key="output" if tools else None
        )

        # Build agent input (text or PDF message)
        if request_body.pdf:
            agent_input = build_pdf_message(
                prompt=request_body.prompt,
                pdf_base64=request_body.pdf,
                pdf_filename=request_body.pdfFilename,
                use_vision=request_body.pdfUseVision or False,
                max_pages=10 if request_body.pdfUseVision else 50
            )
        else:
            agent_input = request_body.prompt

        # Release the DB connection before the (potentially long) LLM call
        chat_session_id = session.id
        db.close()
        print("[AGENT COMPLETION] DB connection released before LLM call")

        user_email = auth.get('email', 'unknown')
        tool_calls = []

        # ── Invoke ────────────────────────────────────────────────────
        if tools:
            if provider == "openai":
                agent_langchain = create_openai_tools_agent(llm, tools, prompt_template)
            else:
                agent_langchain = create_tool_calling_agent(llm, tools, prompt_template)

            agent_executor = AgentExecutor(
                agent=agent_langchain,
                tools=tools,
                memory=memory,
                max_iterations=10
            ).with_config({
                "run_name": "Agent",
                "callbacks": callbacks,
                "metadata": {"user_email": user_email, "agent_id": agent_id, "session_id": str(session_uuid)},
                "tags": [f"user:{user_email}", f"agent:{agent_id}"]
            })

            # Store user message before the call
            _persist_user_message(chat_session_id, request_body.prompt, request_body.pdfFilename)

            result = await agent_executor.ainvoke({"input": agent_input})
            output = result.get("output", "")

            # Collect tool calls from intermediate steps
            for step in result.get("intermediate_steps", []):
                action, observation = step
                formatted = format_tool_call(
                    tool_name=action.tool,
                    tool_input=action.tool_input,
                    tool_output=str(observation)
                )
                if formatted:
                    tool_calls.append(formatted)

        else:
            # Simple chat — no tools
            formatted_input = prompt_template.format_messages(
                chat_history=message_history.messages,
                input=agent_input
            )

            _persist_user_message(chat_session_id, request_body.prompt, request_body.pdfFilename)

            config = {"callbacks": callbacks} if callbacks else {}
            result = await llm.ainvoke(formatted_input, config=config)
            output = result.content

        print(f"[AGENT COMPLETION] Completed — output length: {len(output)} chars")

        # Persist AI response
        _persist_ai_message(chat_session_id, output, tool_calls or None)

        return JSONResponse(content={
            "output": output,
            "tool_calls": tool_calls,
            "session_id": request_body.sessionId,
            "agent_id": agent_id,
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"[AGENT COMPLETION] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Completion failed: {str(e)}"
        )


def _fresh_db():
    """Return a new short-lived DB session for a single write."""
    from src.db.database import SessionLocal
    return SessionLocal()


def _persist_user_message(chat_session_id: int, prompt: str, pdf_filename: Optional[str] = None):
    """Write user message using a short-lived DB session."""
    db = _fresh_db()
    try:
        store_user_message(db, chat_session_id, prompt, pdf_filename)
    finally:
        db.close()


def _persist_ai_message(chat_session_id: int, content: str, tool_calls=None):
    """Write AI message using a short-lived DB session."""
    db = _fresh_db()
    try:
        store_ai_message(db, chat_session_id, content, tool_calls)
    finally:
        db.close()
