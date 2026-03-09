from __future__ import annotations

from typing import AsyncGenerator

from src.routers.swarms.plain_llm import complete_request, stream_request
from src.routers.swarms.policy import build_agent_reply_request, build_router_request, parse_router_decision
from src.routers.swarms.repository import persist_ai_message
from src.routers.swarms.turn_state import encode_turn_state
from src.routers.swarms.types import PreparedTurnContext, RouterDecision, StreamEvent, TurnResult, TurnState


_MAX_RESPONSES_PER_TURN = 5


async def route_turn(context: PreparedTurnContext) -> RouterDecision:
    request = build_router_request(
        history=context.history,
        agents=context.agent_list,
        supervisor_prompt=context.supervisor_prompt,
    )
    context.session_logger.log_llm_messages(request.label, request.messages)
    text = await complete_request(
        provider=context.provider,
        model=context.supervisor_model,
        api_key=context.api_key,
        request=request,
        session_logger=context.session_logger,
    )
    decision = parse_router_decision(text, (agent["name"] for agent in context.agent_list))
    latest_prompt = context.history[-1].content[:80] if context.history else ""
    context.session_logger.log_route(latest_prompt, decision.next_speakers, decision.reason)
    return decision


async def execute_turn(context: PreparedTurnContext) -> TurnResult:
    decision = await route_turn(context)
    if not decision.next_speakers:
        return TurnResult(
            agent_name="",
            content="",
            state_token=None,
            done=True,
            route_reason=decision.reason,
        )

    agent_name = decision.next_speakers[0]
    agent = context.agent_definitions.get(agent_name)
    if not agent:
        return TurnResult(
            agent_name="",
            content="",
            state_token=None,
            done=True,
            route_reason=decision.reason,
        )

    request = build_agent_reply_request(agent=agent, history=context.history)
    context.session_logger.log_llm_messages(request.label, request.messages)
    context.session_logger.log_agent_start(
        agent.name,
        len(context.history),
        context.history[-1].content[:80] if context.history else "",
    )
    started_at = context.session_logger.timer()
    content = await complete_request(
        provider=context.provider,
        model=agent.model,
        api_key=context.api_key,
        request=request,
        session_logger=context.session_logger,
    )
    context.session_logger.log_agent_end(
        agent.name,
        content,
        context.session_logger.timer() - started_at,
    )

    persisted = persist_ai_message(
        context.db,
        chat_session_id=context.chat_session_id,
        content=content,
        agent_name=agent.name,
    )
    response_count = context.state.response_count + 1
    done = response_count >= _MAX_RESPONSES_PER_TURN
    state_token = None
    if not done:
        state_token = encode_turn_state(
            TurnState(
                session_id=context.state.session_id,
                response_count=response_count,
                last_message_id=persisted.id,
                swarm_hash=context.state.swarm_hash,
            )
        )
    return TurnResult(
        agent_name=agent.name,
        content=content,
        state_token=state_token,
        done=done,
        route_reason=decision.reason,
    )


async def stream_turn(context: PreparedTurnContext) -> AsyncGenerator[StreamEvent, None]:
    decision = await route_turn(context)
    if not decision.next_speakers:
        yield StreamEvent(
            event="tts_turn_result",
            data={
                "agentName": "",
                "content": "",
                "stateToken": None,
                "done": True,
                "routeReason": decision.reason,
            },
        )
        return

    agent_name = decision.next_speakers[0]
    agent = context.agent_definitions.get(agent_name)
    if not agent:
        yield StreamEvent(
            event="tts_turn_result",
            data={
                "agentName": "",
                "content": "",
                "stateToken": None,
                "done": True,
                "routeReason": decision.reason,
            },
        )
        return

    request = build_agent_reply_request(agent=agent, history=context.history)
    context.session_logger.log_llm_messages(request.label, request.messages)
    context.session_logger.log_agent_start(
        agent.name,
        len(context.history),
        context.history[-1].content[:80] if context.history else "",
    )
    started_at = context.session_logger.timer()

    yield StreamEvent(event="swarm_agent_start", data={"agentName": agent.name})
    content = ""
    async for chunk in stream_request(
        provider=context.provider,
        model=agent.model,
        api_key=context.api_key,
        request=request,
        session_logger=context.session_logger,
    ):
        content += chunk
        yield StreamEvent(
            event="swarm_chat_model_stream",
            data={"agentName": agent.name, "data": chunk},
        )

    yield StreamEvent(event="swarm_agent_end", data={"agentName": agent.name})
    context.session_logger.log_agent_end(
        agent.name,
        content,
        context.session_logger.timer() - started_at,
    )

    persisted = persist_ai_message(
        context.db,
        chat_session_id=context.chat_session_id,
        content=content,
        agent_name=agent.name,
    )
    response_count = context.state.response_count + 1
    done = response_count >= _MAX_RESPONSES_PER_TURN
    state_token = None
    if not done:
        state_token = encode_turn_state(
            TurnState(
                session_id=context.state.session_id,
                response_count=response_count,
                last_message_id=persisted.id,
                swarm_hash=context.state.swarm_hash,
            )
        )
    yield StreamEvent(
        event="tts_turn_result",
        data={
            "agentName": agent.name,
            "content": content,
            "stateToken": state_token,
            "done": done,
            "routeReason": decision.reason,
        },
    )
