# AI Agent World Game - Treasure Hunt 🎮

A turn-based game where AI agents explore a grid world, move around, collect treasures, and compete for the highest score!

## Game Overview

Three AI agents (Alice, Bob, Charlie) are placed in a 6x6 grid world filled with treasures and obstacles. Each agent uses GPT-4 to decide their moves strategically. The agent with the most points at the end wins!

## Features

- **Grid-Based World**: 6x6 tile-based environment
- **AI-Controlled Agents**: Each agent uses OpenAI's GPT-4 to make decisions
- **Multiple Objects**:
  - 🏆 Treasure (10 points)
  - 💎 Gem (5 points)
  - 🪙 Coin (3 points)
  - 🌳 Tree (obstacle - can't pass)
- **Turn-Based Gameplay**: Agents take turns moving and collecting items
- **Agent Vision**: Each agent sees a 3x3 area around them
- **Strategic AI**: Agents think strategically to maximize their score

## How to Play

### Installation

```bash
# Already installed if you have the multi-agent system
pip install -r requirements_multiagent.txt
```

Make sure your `.env` file has your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

### Run the Game

```bash
python agent_world_game.py
```

The game will:
1. Show the initial world state
2. Each turn, show each agent's view and let them decide their action
3. Update the world based on actions
4. Continue for 15 turns
5. Declare the winner!

## Game Mechanics

### Agent Actions

Agents can perform one action per turn:
- `<move_north>`: Move up
- `<move_south>`: Move down
- `<move_east>`: Move right
- `<move_west>`: Move left

### Agent View

Each agent sees:
- Their current position
- Their score and inventory
- A 3x3 grid around them showing:
  - 👤 = Themselves
  - 🤖 = Other agents
  - Treasures, gems, coins
  - Trees (obstacles)
  - Empty spaces

### Scoring

- Treasure 🏆: 10 points
- Gem 💎: 5 points
- Coin 🪙: 3 points

### Rules

- Agents cannot move through trees 🌳
- Agents cannot move outside the grid boundaries
- When an agent moves onto a tile with treasure, they automatically collect it
- The game runs for 15 turns
- The agent with the highest score wins!

## Example Game Flow

```
Turn 1:
Alice's View:
Position: (0, 0) | Score: 0

Surroundings:
👤 .  .
.  .  .
.  .  .

[Alice] Thinking...
[Alice] Response: <move_east>
✓ Moved east

Turn 2:
Bob's View:
Position: (5, 5) | Score: 0

Surroundings:
.  .  🪙
.  .  .
.  .  👤

[Bob] Thinking...
[Bob] Response: <move_west>
✓ Moved west and found 🪙! (+3 points)
```

## Customization

### Change World Size

```python
world = World(size=8)  # Make it 8x8
```

### Add More Treasures

```python
world.add_object(TileType.TREASURE, (3, 3), 10)
world.add_object(TileType.GEM, (4, 2), 5)
```

### Change Number of Turns

```python
max_turns = 20  # Play for 20 turns instead of 15
```

### Use Different Models

```python
ais = {
    "Alice": GameAI("Alice", client, model="gpt-4o"),
    "Bob": GameAI("Bob", client, model="gpt-4-turbo"),
}
```

## Future Extensions

Potential features to add:
- **Cooperative mode**: Agents work together to maximize total score
- **Special abilities**: Each agent has unique powers
- **Dynamic obstacles**: Trees that move or grow
- **Agent communication**: Agents can send messages to each other
- **More complex actions**: Push objects, build, trade items
- **Fog of war**: Agents can only see areas they've explored
- **Multiplayer**: Human players compete against AI agents

## Architecture

- **World**: Manages the grid, objects, and agent positions
- **GameObject**: Represents items in the world (treasures, obstacles)
- **GameAgent**: Tracks agent state (position, score, inventory)
- **GameAI**: AI controller that uses GPT-4 to make decisions
- **Game Loop**: Turn-based system that updates world state

## Why This Is Interesting

This game demonstrates:
1. **AI Spatial Reasoning**: How LLMs understand and navigate 2D space
2. **Strategic Planning**: LLMs making decisions based on incomplete information
3. **Multi-Agent Competition**: Multiple AIs competing in the same environment
4. **Embodied AI**: Language models controlling agents in a simulated world
5. **Emergent Behavior**: Unexpected strategies and interactions between agents

Enjoy watching AI agents explore, compete, and strategize! 🎮
