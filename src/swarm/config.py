"""
Parse agent config for hierarchical swarm (multiAgentArchitecture: hierarchicalSwarm).
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class WorkerSpec:
    agent_name: str
    agent_description: str
    system_prompt: str
    model_name: str


@dataclass
class DirectorSpec:
    name: str
    model_name: str
    system_prompt: Optional[str] = None


@dataclass
class SwarmConfig:
    director: DirectorSpec
    workers: List[WorkerSpec]
    max_loops: int = 1


def parse_swarm_config(agent_config: Dict[str, Any]) -> Optional[SwarmConfig]:
    """
    Extract swarm config from agent config if this agent is a hierarchical swarm.
    Expects config.data.multiAgentArchitecture === "hierarchicalSwarm" and config.data.swarm.
    """
    if not agent_config:
        return None
    data = agent_config.get("data") or {}
    if data.get("multiAgentArchitecture") != "hierarchicalSwarm":
        return None
    swarm_data = data.get("swarm")
    if not swarm_data or not isinstance(swarm_data, dict):
        return None
    director_data = swarm_data.get("director") or {}
    workers_data = swarm_data.get("workers") or []
    if not workers_data:
        return None
    director = DirectorSpec(
        name=director_data.get("name") or "Director",
        model_name=director_data.get("modelName") or "gpt-4o-mini",
        system_prompt=director_data.get("systemPrompt"),
    )
    workers: List[WorkerSpec] = []
    for w in workers_data:
        if not isinstance(w, dict) or not w.get("agentName"):
            continue
        workers.append(
            WorkerSpec(
                agent_name=w["agentName"],
                agent_description=w.get("agentDescription") or "",
                system_prompt=w.get("systemPrompt") or f"You are {w['agentName']}.",
                model_name=w.get("modelName") or "gpt-4o-mini",
            )
        )
    if not workers:
        return None
    return SwarmConfig(
        director=director,
        workers=workers,
        max_loops=int(swarm_data.get("maxLoops") or 1),
    )
