import asyncio
from typing import List

from .agents import Agent
from .environment import GuessPhraseEnv
from .ui import RLConfig, gather_config

async def run_cycle(env: GuessPhraseEnv, agents: List[Agent], cycle: int) -> List[str]:
    streams = [agent.generate_stream(env.target) for agent in agents]
    tasks = [asyncio.create_task(stream.__anext__()) for stream in streams]
    tokens = [[] for _ in agents]
    while tasks:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for i, t in enumerate(tasks):
            if t in done:
                try:
                    token = t.result()
                    print(f"<{agents[i].name}>\n{token}\n<{agents[i].name}/>\n")
                    tokens[i].append(token)
                    tasks[i] = asyncio.create_task(streams[i].__anext__())
                except StopAsyncIteration:
                    tasks.remove(t)
                    streams[i] = None
                    tasks[i] = None
        tasks = [t for t in tasks if t]
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
    for c in range(cycles):
        await run_cycle(env, agents, c+1)

if __name__ == "__main__":
    asyncio.run(main())
