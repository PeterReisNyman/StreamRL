from dataclasses import dataclass
from typing import List
import random
import string

def random_name(length: int = 6) -> str:
    """Generate a random lowercase name of given length."""
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))

@dataclass
class AgentConfig:
    name: str
    system_message: str

def gather_config(num_agents: int) -> List[AgentConfig]:
    agents = []
    for i in range(num_agents):
        name = random_name()
        sys_msg = input(f"System message for {name}: ") or ""
        agents.append(AgentConfig(name, sys_msg))
    return agents

@dataclass
class RLConfig:
    task: str
    cycles: int
    agents: List[AgentConfig]
