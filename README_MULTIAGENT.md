# Multi-Agent Streaming System for Collaborative Problem Solving

A token-level interleaved multi-agent system where AI agents collaborate to solve problems by streaming their thoughts and solutions in real-time with different visibility modes.

## Key Application: Collaborative Problem Solving

This system enables multiple specialized AI agents to work together on complex problems:
- **Creative Agent**: Generates innovative ideas and approaches
- **Analytical Agent**: Evaluates feasibility and identifies issues
- **Practical Agent**: Focuses on implementation and actionable steps

Each agent sees the others' tokens **as they're generated** and can respond immediately, creating tight feedback loops and richer collaboration.

## Features

- **Token-Level Interleaving**: Agents take turns generating one token at a time, seeing all previous tokens before generating their next one
- **True Context Sharing**: Each agent sees the full conversation history before each token generation
- **Two Visibility Modes**:
  - `<broadcast>` 📢: Tokens visible to all agents (public discussion)
  - `<internal>` 🤔: Tokens only visible to the agent itself (private thinking/brainstorming)
- **Real-time Collaboration**: Agents react to partial information, not just complete thoughts
- **Specialized Roles**: Each agent brings unique perspective to problem-solving

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

The demo showcases collaborative problem solving with three agents tackling a real-world challenge:

```bash
python multi_agent_stream.py
```

**Demo Problem**: How can a small town library increase youth engagement by 50% with a $5,000 budget?

Watch as:
- **Alice** (Creative) brainstorms innovative ideas using `<internal>` mode, then shares via `<broadcast>`
- **Bob** (Analytical) evaluates feasibility and identifies potential issues
- **Charlie** (Practical) provides concrete implementation steps and budget considerations

You'll see token-level interleaving in action and how agents build on each other's ideas!

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

### Primary: Collaborative Problem Solving
- **Business Strategy**: Creative, analytical, and practical agents collaborate on business challenges
- **Product Design**: Innovation, technical feasibility, and user experience perspectives
- **System Optimization**: Multiple viewpoints on complex system improvements
- **Research Questions**: Interdisciplinary collaboration with different methodological approaches

### Other Applications
- **Multi-agent debates**: Agents argue positions with private strategy planning
- **Educational simulations**: Demonstrate how different perspectives interact
- **Creative writing**: Multiple authors building narratives with private plotting
- **Game theory research**: Study strategic interactions with hidden information
