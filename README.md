# StreamRL - Multi-Agent AI Systems 🤖

A collection of cutting-edge multi-agent AI systems featuring token-level interleaving, collaborative problem solving, and embodied AI in 3D worlds.

## 🎯 Projects

### 1. **Token-Streaming Multi-Agent System**
📁 [`multi_agent_stream.py`](./multi_agent_stream.py) | 📖 [README](./README_MULTIAGENT.md)

The core multi-agent communication system with token-level interleaving.

**Features:**
- ✅ Real-time token-by-token streaming
- ✅ Two visibility modes: `<broadcast>` and `<internal>`
- ✅ True context sharing (agents see all previous tokens)
- ✅ Specialized agent roles (Creative, Analytical, Practical)

**Use Case:** Collaborative problem solving where multiple perspectives tackle complex challenges.

```bash
python multi_agent_stream.py
```

---

### 2. **AI Agent Treasure Hunt Game** 🎮
📁 [`agent_world_game.py`](./agent_world_game.py) | 📖 [README](./README_GAME.md)

A 2D grid-based treasure hunt where AI agents compete to collect items.

**Features:**
- ✅ 6x6 tile-based world
- ✅ AI agents navigate and collect treasures
- ✅ Strategic decision-making
- ✅ Turn-based gameplay

**Use Case:** Demonstrating AI spatial reasoning and competitive behavior in a simple environment.

```bash
python agent_world_game.py
```

---

### 3. **AI Agents in VoxelCraft** 🏗️🤖
📁 [`voxelcraft_ai_controller.py`](./voxelcraft_ai_controller.py) | 📖 [README](./README_VOXELCRAFT_AI.md)

**🌟 THE MAIN EVENT** - AI agents that see, think, communicate, and build in a 3D Minecraft-like world!

**Features:**
- ✅ **3D Embodied AI**: Agents move, look, and build in VoxelCraft
- ✅ **Vision**: Agents perceive their position, other players, and world state
- ✅ **Communication**: Agents use `<broadcast>` and `<internal>` to coordinate
- ✅ **Building**: Place/break blocks (grass, stone, wood)
- ✅ **Collaborative Tasks**: Multiple agents work on different goals simultaneously

**Example Tasks:**
- Alice: Build a stone tower
- Bob: Build a wooden house
- Charlie: Create a grass garden

**Quick Start:**
```bash
# Terminal 1: Start VoxelCraft
python run_voxelcraft.py

# Terminal 2: Run AI agents
python voxelcraft_ai_controller.py

# Or use the quick start script:
./start_voxelcraft_demo.sh
```

**Watch them build:**
```
http://127.0.0.1:8000/index.html?room=ai_agents&name=Observer
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements_multiagent.txt

# Set up .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Run the Main Demo (VoxelCraft AI)

```bash
./start_voxelcraft_demo.sh
```

This will:
1. Start VoxelCraft server
2. Open browser to watch the agents
3. Run AI agents who will build structures
4. You'll see Alice, Bob, and Charlie building in real-time!

## 📊 Comparison Table

| Feature | Multi-Agent Stream | Treasure Hunt | **VoxelCraft AI** |
|---------|-------------------|---------------|-------------------|
| **Dimension** | Text-based | 2D Grid | **3D World** |
| **Agent Vision** | Text messages | 3x3 local view | **Full world state** |
| **Actions** | Thinking/Communication | 4 directions | **12+ actions (move, build, rotate)** |
| **Persistence** | Conversation history | Scores | **Permanent structures** |
| **Communication** | Token-streaming | None | **Token-streaming** |
| **Visualization** | Console output | ASCII grid | **3D Graphics** |
| **Complexity** | Medium | Low | **High** |
| **Coolness Factor** | 8/10 | 6/10 | **10/10** ✨ |

## 🎓 Research Applications

These systems enable research in:

1. **Embodied AI**: How language models control agents in physical(ish) spaces
2. **Multi-Agent Coordination**: Communication protocols between AI agents
3. **Spatial Reasoning**: Can LLMs navigate and build in 2D/3D?
4. **Emergent Behavior**: Unexpected strategies and interactions
5. **Task Decomposition**: Breaking complex goals into actions
6. **Tool Use**: Using blocks, movement, and communication as tools
7. **Collaborative Problem Solving**: Multiple perspectives tackling challenges

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│          OpenAI GPT-4 Models                │
│  (Decision Making & Communication)          │
└──────────────┬──────────────────────────────┘
               │
               ↓
┌─────────────────────────────────────────────┐
│     Token-Streaming Multi-Agent System      │
│  • <broadcast> / <internal> modes           │
│  • Message routing                          │
│  • Context management                       │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
       ↓               ↓
┌─────────────┐  ┌──────────────────┐
│ 2D Grid     │  │  VoxelCraft 3D   │
│ Game World  │  │  Voxel Engine    │
└─────────────┘  └──────────────────┘
```

## 📖 Documentation

- **[Multi-Agent System](./README_MULTIAGENT.md)** - Core communication system
- **[Treasure Hunt Game](./README_GAME.md)** - 2D grid game
- **[VoxelCraft AI](./README_VOXELCRAFT_AI.md)** - 3D embodied AI agents
- **[VoxelCraft](./README_GAME.md)** - Original voxel engine

## 🎥 Example Output

### Multi-Agent Problem Solving
```
[Alice]📢 'Create' [EXTRA: 3 tokens: 'gaming tournaments with']
[Bob]📢 'Sounds' [EXTRA: 2 tokens: 'feasible but']
[Charlie]📢 'Budget' [EXTRA: 1 token: 'considerations:']
```

### VoxelCraft AI Agents
```
[Alice] Thinking...
[Alice] Response: <broadcast> Building my tower at (0, 120, 0)!
Placed STONE at (0, 120, 0)
[Alice] ✓ Executed PLACE_BLOCK

[Bob] Thinking...
[Bob] Context includes: [Alice]: Building my tower at (0, 120, 0)!
[Bob] Response: <broadcast> Great! Starting house nearby. <move_forward>
[Bob] ✓ Executed MOVE_FORWARD
```

## 🔧 Advanced Usage

### Custom Tasks in VoxelCraft

```python
alice.set_task("Build a pyramid with 5 layers")
bob.set_task("Create a path from Alice's pyramid to Charlie's garden")
charlie.set_task("Build a wall around the entire area")
```

### Add More Agents

```python
david = VoxelAgent("David", client, david_client, color="#ffff00")
david.set_task("Build a castle")
agents.append(david)
```

### Change Models

```python
alice = VoxelAgent("Alice", client, alice_client, model="gpt-4o")
bob = VoxelAgent("Bob", client, bob_client, model="gpt-4-turbo")
```

## 🐛 Troubleshooting

### VoxelCraft not connecting
```bash
# Check if server is running
curl http://127.0.0.1:8000/snapshot?room=ai_agents

# Restart server
python run_voxelcraft.py
```

### Agents not appearing
- Refresh browser
- Check room name matches (`ai_agents`)
- Look at console for errors

### API Rate Limits
If you hit rate limits, reduce turns or add delays:
```python
max_turns = 10  # Fewer turns
time.sleep(2)   # Longer delays between actions
```

## 🌟 Why This Is Special

This is one of the first systems that combines:
1. **Language models as embodied agents** in a 3D world
2. **Token-level interleaving** for tight coordination
3. **Real-time communication** between multiple AI agents
4. **Persistent 3D structures** that you can see and interact with
5. **Private/public thinking** modes for strategic behavior

You're literally watching GPT-4 agents **build things** while **talking to each other**! 🤯

## 📝 Citation

If you use this for research, please cite:

```bibtex
@software{streamrl_2024,
  title={StreamRL: Token-Streaming Multi-Agent Systems with Embodied AI},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/StreamRL}
}
```

## 📜 License

MIT License - feel free to use for research or fun!

## 🤝 Contributing

Contributions welcome! Some ideas:
- Add vision (screenshot → GPT-4V)
- More complex building algorithms
- Competitive games between agents
- Teaching scenarios
- Natural language commands from humans

## 🙏 Acknowledgments

- OpenAI for GPT-4
- VoxelCraft voxel engine
- The multi-agent AI research community

---

**Made with ❤️ and 🤖 by AI agents for AI agents**

*Now go watch your AI agents build something amazing!* 🏗️✨
