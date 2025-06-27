from dataclasses import dataclass
from typing import List

@dataclass
class AgentConfig:
    name: str
    system_message: str

def gather_config(num_agents: int) -> List[AgentConfig]:
    agents = []
    for i in range(num_agents):
        name = input(f"Agent {i+1} name: ") or f"agent{i+1}"
        sys_msg = input(f"System message for {name}: ") or ""
        agents.append(AgentConfig(name, sys_msg))
    return agents

@dataclass
class RLConfig:
    task: str
    cycles: int
    agents: List[AgentConfig]
