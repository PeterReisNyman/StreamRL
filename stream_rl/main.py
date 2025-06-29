import asyncio
import json
from typing import List, Dict

from .agents import Agent
from .environment import GuessPhraseEnv
from .ui import RLConfig, gather_config


def add_message(messages: List[Dict[str, str]], agent_name: str, content: str) -> None:
    """Add a message and convert previous messages from this agent to assistant."""
    for m in messages:
        if m.get("speaker") == agent_name:
            m["role"] = "assistant"
    messages.append({"role": agent_name, "speaker": agent_name, "content": content})


def print_messages(messages: List[Dict[str, str]]) -> None:
    display = [{"role": m["role"], "content": m["content"]} for m in messages]
    print("messages:")
    print(json.dumps(display, indent=4))

async def run_cycle(env: GuessPhraseEnv, agents: List[Agent], cycle: int, messages: List[Dict[str, str]]) -> List[str]:
    streams = [agent.generate_stream(env.target) for agent in agents]
    tasks = {i: asyncio.create_task(stream.__anext__()) for i, stream in enumerate(streams)}
    tokens = [[] for _ in agents]
    while tasks:
        done, _ = await asyncio.wait(tasks.values(), return_when=asyncio.FIRST_COMPLETED)
        for i, t in list(tasks.items()):
            if t in done:
                try:
                    token = t.result()
                    add_message(messages, agents[i].name, token)
                    print_messages(messages)
                    tokens[i].append(token)
                    tasks[i] = asyncio.create_task(streams[i].__anext__())
                except StopAsyncIteration:
                    del tasks[i]
                    streams[i] = None
    scores = [env.score(tok) for tok in tokens]
    winner_idx = scores.index(max(scores))
    winner_tokens = tokens[winner_idx]
    print(f"Cycle {cycle} winner: {agents[winner_idx].name} with score {scores[winner_idx]}")
    for agent in agents:
        agent.train(winner_tokens)
    return winner_tokens

async def main():
    num_agents = int(input("Number of agents: ") or 2)
    agents_cfg = gather_config(num_agents)
    task = input("Task / target phrase: ") or "hello"
    cycles = int(input("Number of cycles: ") or 3)
    agents = [Agent(cfg.name, cfg.system_message) for cfg in agents_cfg]
    env = GuessPhraseEnv(task)
    messages = []
    for c in range(cycles):
        await run_cycle(env, agents, c+1, messages)

if __name__ == "__main__":
    asyncio.run(main())
