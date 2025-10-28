# AI Agents in VoxelCraft 🤖🎮

AI agents that can see, think, communicate, and build in a 3D Minecraft-like world!

## Overview

This system connects **GPT-4 powered AI agents** to **VoxelCraft**, enabling them to:
- 🏗️ **Build structures** collaboratively
- 👀 **Perceive the 3D world** (position, other players, blocks)
- 🧠 **Think strategically** using `<internal>` mode
- 💬 **Communicate** with other agents using `<broadcast>` mode
- 🚶 **Move and navigate** in 3D space
- 🎯 **Complete tasks** like building towers, houses, gardens

## Features

### AI Agent Capabilities

Each AI agent can:
- **Move**: Forward, backward, left, right, up, down
- **Rotate**: Turn left/right, look up/down
- **Build**: Place grass, stone, wood blocks
- **Break**: Remove blocks
- **Communicate**: Share thoughts with other agents or think privately
- **Perceive**: See their position, rotation, other players

### Token-Level Interleaving

Agents use the same token-streaming system as the collaborative problem solver:
- `<broadcast>`: Share thoughts with all agents
- `<internal>`: Private thinking (other agents can't see)
- Agents take turns deciding actions
- Messages are shared in real-time between agents

## Installation

Already installed if you have the multi-agent system:
```bash
pip install -r requirements_multiagent.txt
```

## Usage

### Step 1: Start VoxelCraft Server

In one terminal:
```bash
python3 run_voxelcraft.py
```

This opens VoxelCraft in your browser. Keep this running.

### Step 2: Run AI Agents

In another terminal:
```bash
python3 voxelcraft_ai_controller.py
```

### Step 3: Watch Them Build!

Open your browser to see the agents in action:
```
http://127.0.0.1:8000/index.html?room=ai_agents&name=Observer
```

You'll see:
- **Alice** (red) building a stone tower
- **Bob** (green) building a wooden house
- **Charlie** (blue) creating a grass garden

## How It Works

### Architecture

```
┌─────────────────┐
│   GPT-4 Agent   │
│     (Alice)     │
└────────┬────────┘
         │
         ↓ Decisions
┌─────────────────┐
│  VoxelCraft     │
│     Client      │ ←→ HTTP/SSE
└────────┬────────┘
         │
         ↓ Actions
┌─────────────────┐
│  VoxelCraft     │
│     Server      │
└────────┬────────┘
         │
         ↓ Updates
┌─────────────────┐
│   3D World      │
│   (Browser)     │
└─────────────────┘
```

### Agent Decision Loop

Each turn, an agent:
1. **Perceives** the world (position, other players, blocks placed)
2. **Thinks** using GPT-4 about what to do next
3. **Decides** on an action (move, build, communicate)
4. **Executes** the action in VoxelCraft
5. **Broadcasts** position update to server

### Communication Between Agents

Agents can communicate while building:
```
Alice (internal): "I should build a tower here"
Alice (broadcast): "Starting my tower at (0, 120, 0)"
Bob (internal): "I'll build nearby"
Bob (broadcast): "Building house at (10, 120, 0)"
```

### Available Actions

**Movement:**
- `<move_forward>`: Move 2 blocks forward
- `<move_backward>`: Move 2 blocks backward
- `<move_left>`: Strafe left 2 blocks
- `<move_right>`: Strafe right 2 blocks
- `<move_up>`: Fly up 2 blocks
- `<move_down>`: Fly down 2 blocks

**Rotation:**
- `<turn_left>`: Turn 45° left
- `<turn_right>`: Turn 45° right
- `<look_up>`: Tilt camera up
- `<look_down>`: Tilt camera down

**Building:**
- `<place_grass>`: Place grass block ahead
- `<place_stone>`: Place stone block ahead
- `<place_wood>`: Place wood block ahead
- `<break_block>`: Break block ahead

**Communication:**
- `<broadcast>`: Share thoughts with all agents
- `<internal>`: Think privately
- `<wait>`: Do nothing this turn

## Example Agent Behavior

### Turn 1: Alice
```
[Alice] Thinking...
[Alice] Response: <internal> I need to build a tower. Let me start by moving to a good location...
[Alice] 🔄 Switched to INTERNAL
[Alice] ✓ Executed WAIT
```

### Turn 2: Alice
```
[Alice] Thinking...
[Alice] Response: <broadcast> Starting my stone tower now! <place_stone>
[Alice] 🔄 Switched to BROADCAST
Placed STONE at (0, 120, 0)
[Alice] ✓ Executed PLACE_BLOCK
```

### Turn 3: Bob (sees Alice's message)
```
[Bob] Thinking...
Context includes: [Alice]: Starting my stone tower now!
[Bob] Response: <broadcast> Great! I'll build a house nearby. <move_forward>
[Bob] ✓ Executed MOVE_FORWARD
```

## Customization

### Change Agent Tasks

```python
alice.set_task("Build a pyramid out of stone")
bob.set_task("Create a bridge connecting two points")
charlie.set_task("Build a farm with different crops")
```

### Add More Agents

```python
david_client = VoxelCraftClient(room=room)
david_client.position = VoxelPosition(20, 120, 0, 0, 0)
david = VoxelAgent("David", client, david_client, color="#ffff00")
david.set_task("Build a castle")
agents.append(david)
```

### Change Number of Turns

```python
max_turns = 50  # Let them build for longer
```

### Different Starting Positions

```python
alice_client.position = VoxelPosition(
    x=0,      # X coordinate
    y=120,    # Y coordinate (height)
    z=0,      # Z coordinate
    yaw=0,    # Rotation (radians)
    pitch=0   # Look up/down (radians)
)
```

### Use Different Models

```python
alice = VoxelAgent("Alice", client, alice_client, model="gpt-4o")
bob = VoxelAgent("Bob", client, bob_client, model="gpt-4-turbo")
```

## Advanced: Custom Room

Create a private room for your agents:
```python
room = "my_custom_room"
alice_client = VoxelCraftClient(room=room)
```

Then view at:
```
http://127.0.0.1:8000/index.html?room=my_custom_room&name=Observer
```

## Troubleshooting

### "Connection refused" error
Make sure VoxelCraft server is running:
```bash
python run_voxelcraft.py
```

### Agents not appearing in browser
- Check the room name matches
- Refresh the browser page
- Check console for errors

### Agents placing blocks in wrong location
- The coordinate system: X/Z are horizontal, Y is vertical
- Blocks are placed 3 units ahead of agent
- Agent needs to be facing the right direction

## Research Applications

This system enables research in:

1. **Embodied AI**: Language models controlling agents in 3D space
2. **Multi-Agent Coordination**: How AI agents collaborate on building tasks
3. **Spatial Reasoning**: Can LLMs navigate and build in 3D?
4. **Communication Protocols**: How agents coordinate via language
5. **Emergent Behavior**: Unexpected strategies and interactions
6. **Task Planning**: Breaking down "build a house" into actions
7. **Tool Use**: Using building blocks as tools to achieve goals

## Future Extensions

- **Vision Integration**: Feed screenshots to GPT-4V for visual understanding
- **More Complex Tasks**: Build specific structures, mazes, art
- **Competitive Games**: Agents compete for resources or territory
- **Teaching**: One agent teaches another how to build
- **Dynamic Goals**: Goals change based on what agents see
- **Persistent Worlds**: Save/load world state across sessions
- **Natural Language Commands**: Tell agents what to build via chat

## Why This Is Cool

This is **embodied AI** in action! Language models that:
- Actually move and act in a 3D world
- Make spatial decisions (where to build, how to navigate)
- Collaborate with other AI agents in real-time
- Build persistent structures you can see
- Communicate naturally while executing tasks

Watch your AI agents become builders! 🏗️🤖

---

**Built on:**
- VoxelCraft (3D voxel engine)
- OpenAI GPT-4 (decision making)
- Token-streaming multi-agent system (communication)
