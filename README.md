# StreamRL

StreamRL is a simple demonstration of a streaming multi-agent reinforcement loop.
Agents generate tokens in parallel to solve a task. After each cycle, the best
agent's output is used as synthetic training data for the next round.

Run the interactive demo:

```bash
python -m stream_rl.main
```

You will be prompted for the number of agents, their system messages, the target
phrase and number of cycles. The agents then stream their tokens one by one in
the format:

```
<agent_name>
TOKEN
<agent_name/>
```

After each cycle the agent with the highest score (matching tokens) is selected
as the winner and all agents train on the winner's tokens.
