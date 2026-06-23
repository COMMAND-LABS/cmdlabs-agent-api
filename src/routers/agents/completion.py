"""Agent completion endpoint — non-streaming.

Runs the agent to completion and returns the full output as a single JSON
response.  Uses the same agent setup logic as stream.py via the shared
``prepare_agent_context`` helper.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.core.schemas.ChatSessionPrompt import ChatSessionPrompt
from src.deps import auth_dependency, db_dependency
from src.ratelimit import limiter
from src.routers.agents.context import (
    AgentSetupError,
    persist_ai_message,
    persist_user_message,
    prepare_agent_context,
)
from src.routers.agents.helpers import format_tool_call
from src.utils.langsmith import get_langsmith_callbacks

logger = logging.getLogger(__name__)

router = APIRouter()
callbacks = get_langsmith_callbacks("dynamic-agent-completion")


@router.post("/{agent_id}/completion")
@limiter.limit("60/minute")
async def agent_completion(
    agent_id: int,
    request_body: ChatSessionPrompt,
    db: db_dependency,
    auth: auth_dependency,
    request: Request,
):
    """Run an agent to completion and return the full output as JSON."""
    try:
        ctx = await prepare_agent_context(
            agent_id=agent_id,
            session_id=request_body.sessionId,
            prompt=request_body.prompt,
            db=db,
            auth=auth,
            request=request,
            callbacks=callbacks,
            streaming=False,
            pdf_base64=request_body.pdf,
            pdf_filename=request_body.pdfFilename,
            pdf_use_vision=request_body.pdfUseVision or False,
        )
    except AgentSetupError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.detail) from exc

    try:
        tool_calls = []

        persist_user_message(ctx.chat_session_id, ctx.prompt, ctx.pdf_filename)

        if ctx.agent_executor:
            result = await ctx.agent_executor.ainvoke({"input": ctx.agent_input})
            output = result.get("output", "")

            for step in result.get("intermediate_steps", []):
                action, observation = step
                formatted = format_tool_call(
                    tool_name=action.tool,
                    tool_input=action.tool_input,
                    tool_output=str(observation),
                )
                if formatted:
                    tool_calls.append(formatted)
        else:
            formatted_input = ctx.prompt_template.format_messages(
                chat_history=ctx.message_history.messages,
                input=ctx.agent_input,
            )
            config = {"callbacks": ctx.callbacks} if ctx.callbacks else {}
            result = await ctx.llm.ainvoke(formatted_input, config=config)
            output = result.content

        persist_ai_message(ctx.chat_session_id, output, tool_calls or None)

        return JSONResponse(content={
            "output": output,
            "tool_calls": tool_calls,
            "session_id": request_body.sessionId,
            "agent_id": agent_id,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[COMPLETION] Unhandled error during completion: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Completion failed: {e!s}",
        ) from e
