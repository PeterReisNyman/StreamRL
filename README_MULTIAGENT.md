# Multi-Agent Streaming System

A system where multiple AI agents stream tokens and can communicate with different visibility modes.

## Features

- **Multiple AI Agents**: Each agent has a unique name and can stream responses
- **Two Visibility Modes**:
  - `<broadcast>` 📢: Tokens visible to all agents
  - `<internal>` 🤔: Tokens only visible to the agent itself (private thinking)
- **Real-time Token Streaming**: See each token as it's generated
- **Message Routing**: Agents only see messages they're allowed to see
- **Tool System**: Agents can switch modes mid-stream

## Installation

```bash
pip install -r requirements_multiagent.txt
```

Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Basic Example

```python
from multi_agent_stream import MultiAgentSystem
import asyncio

async def main():
    system = MultiAgentSystem(api_key="your-key", model="gpt-4")

    # Add agents
    alice = system.add_agent("Alice")
    bob = system.add_agent("Bob")

    # Run agents
    await system.run_agent("Alice", "Hello, introduce yourself!")
    await system.run_agent("Bob", "Respond to Alice")

    # Get conversation summary
    print(system.get_conversation_summary())

asyncio.run(main())
```

### Running the Demo

```bash
python multi_agent_stream.py
```

## How It Works

### Agent Tools

Each agent has access to two tools:

1. **<broadcast>**: Makes subsequent tokens visible to all agents
   - Use in stream: `<broadcast>`

2. **<internal>**: Makes subsequent tokens private
   - Use in stream: `<internal>`

### Message Visibility

- **Broadcast messages** (📢): All agents receive these in their context
- **Internal messages** (🔒): Only the originating agent sees these

### Example Flow

```
Alice (broadcast): "Hello everyone!"
  → Bob sees this
  → Charlie sees this

Bob (internal): "Hmm, I should think about this..."
  → Only Bob sees this

Bob (broadcast): "Great idea Alice!"
  → Alice sees this
  → Charlie sees this
```

## Architecture

- `MultiAgentSystem`: Manages all agents and message routing
- `Agent`: Individual AI agent with streaming capabilities
- `Message`: Contains content, sender, and visibility rules
- `StreamMode`: Enum defining BROADCAST and INTERNAL modes

## Customization

### Add Custom Agent Behaviors

```python
await system.run_agent(
    "Alice",
    "Your prompt here",
    system_context="You are a creative thinker who loves poetry."
)
```

### Change Models

```python
system = MultiAgentSystem(api_key="your-key", model="gpt-4o")
```

### Access Agent Message History

```python
alice = system.agents["Alice"]
visible_messages = alice.get_visible_messages()
```

## Use Cases

- Multi-agent debates and discussions
- Collaborative problem solving with private reasoning
- AI roleplay scenarios with hidden thoughts
- Research simulations with information asymmetry
