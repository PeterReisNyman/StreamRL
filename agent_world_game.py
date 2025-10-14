"""
AI Agent World Game - Treasure Hunt
Agents move around a grid world, interact with objects, and compete/cooperate
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import random

load_dotenv()


class TileType(Enum):
    EMPTY = "."
    TREE = "🌳"
    TREASURE = "🏆"
    GEM = "💎"
    COIN = "🪙"


@dataclass
class GameObject:
    type: TileType
    value: int
    position: Tuple[int, int]


@dataclass
class GameAgent:
    name: str
    position: Tuple[int, int]
    score: int = 0
    inventory: List[str] = None

    def __post_init__(self):
        if self.inventory is None:
            self.inventory = []


class World:
    def __init__(self, size: int = 6):
        self.size = size
        self.grid = [[TileType.EMPTY for _ in range(size)] for _ in range(size)]
        self.objects: Dict[Tuple[int, int], GameObject] = {}
        self.agents: Dict[str, GameAgent] = {}
        self.turn_count = 0

    def add_agent(self, name: str, position: Tuple[int, int]):
        """Add an agent to the world"""
        self.agents[name] = GameAgent(name, position)

    def add_object(self, obj_type: TileType, position: Tuple[int, int], value: int):
        """Add an object to the world"""
        if position not in self.objects:
            self.objects[position] = GameObject(obj_type, value, position)
            self.grid[position[0]][position[1]] = obj_type

    def can_move_to(self, position: Tuple[int, int]) -> bool:
        """Check if position is valid and not blocked"""
        x, y = position
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        if self.grid[x][y] == TileType.TREE:
            return False
        return True

    def move_agent(self, agent_name: str, direction: str) -> Tuple[bool, str]:
        """Move agent in direction. Returns (success, message)"""
        if agent_name not in self.agents:
            return False, "Agent not found"

        agent = self.agents[agent_name]
        x, y = agent.position

        # Calculate new position
        new_pos = None
        if direction == "north":
            new_pos = (x - 1, y)
        elif direction == "south":
            new_pos = (x + 1, y)
        elif direction == "east":
            new_pos = (x, y + 1)
        elif direction == "west":
            new_pos = (x, y - 1)
        else:
            return False, f"Invalid direction: {direction}"

        # Check if move is valid
        if not self.can_move_to(new_pos):
            return False, f"Cannot move {direction} - blocked or out of bounds"

        # Move agent
        agent.position = new_pos

        # Check for objects at new position
        if new_pos in self.objects:
            obj = self.objects[new_pos]
            agent.score += obj.value
            agent.inventory.append(obj.type.value)
            del self.objects[new_pos]
            self.grid[new_pos[0]][new_pos[1]] = TileType.EMPTY
            return True, f"Moved {direction} and found {obj.type.value}! (+{obj.value} points)"

        return True, f"Moved {direction}"

    def get_agent_view(self, agent_name: str) -> str:
        """Get what the agent can see around them"""
        if agent_name not in self.agents:
            return "Agent not found"

        agent = self.agents[agent_name]
        x, y = agent.position

        view = f"\n=== {agent_name}'s View (Turn {self.turn_count}) ===\n"
        view += f"Position: ({x}, {y}) | Score: {agent.score} | Inventory: {agent.inventory}\n\n"

        # Show 3x3 area around agent
        view += "Surroundings:\n"
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy

                # Check bounds
                if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
                    view += "  "
                    continue

                # Check if agent is here
                if (nx, ny) == agent.position:
                    view += "👤 "
                    continue

                # Check if other agent is here
                other_agent = None
                for other_name, other in self.agents.items():
                    if other_name != agent_name and other.position == (nx, ny):
                        other_agent = other_name
                        break

                if other_agent:
                    view += "🤖 "
                    continue

                # Show tile
                view += self.grid[nx][ny].value + " "

            view += "\n"

        view += "\n"
        return view

    def render(self) -> str:
        """Render the full world"""
        result = f"\n{'='*40}\n"
        result += f"WORLD STATE - Turn {self.turn_count}\n"
        result += f"{'='*40}\n\n"

        for x in range(self.size):
            for y in range(self.size):
                # Check if agent is here
                agent_here = None
                for name, agent in self.agents.items():
                    if agent.position == (x, y):
                        agent_here = name[0].upper()
                        break

                if agent_here:
                    result += f"{agent_here}  "
                else:
                    result += self.grid[x][y].value + " "
            result += "\n"

        result += "\nScores:\n"
        for name, agent in self.agents.items():
            result += f"  {name}: {agent.score} points\n"

        result += f"{'='*40}\n"
        return result


class GameAI:
    """AI agent that plays the game"""

    def __init__(self, name: str, client: OpenAI, model: str = "gpt-4"):
        self.name = name
        self.client = client
        self.model = model
        self.action_history: List[str] = []

    def decide_action(self, world_view: str, game_state: str) -> Tuple[Optional[str], Optional[str]]:
        """Agent decides what action to take. Returns (action, direction/target)"""

        prompt = f"""{world_view}

{game_state}

You are playing a treasure hunt game. Your goal is to collect treasures and score points!

Available actions:
- <move_north>: Move one tile north (up)
- <move_south>: Move one tile south (down)
- <move_east>: Move one tile east (right)
- <move_west>: Move one tile west (left)

Legend:
- 👤 = You
- 🤖 = Other agent
- 🏆 = Treasure (10 points)
- 💎 = Gem (5 points)
- 🪙 = Coin (3 points)
- 🌳 = Tree (can't pass)
- . = Empty space

Think about your strategy, then choose ONE action. Output ONLY the action tag, nothing else.
"""

        messages = [
            {"role": "system", "content": f"You are {self.name}, a competitive treasure hunter. Be strategic and try to maximize your score."},
            {"role": "user", "content": prompt}
        ]

        print(f"\n[{self.name}] Thinking...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )

        content = response.choices[0].message.content.strip()
        print(f"[{self.name}] Response: {content}")

        # Parse action
        if "<move_north>" in content:
            return "move", "north"
        elif "<move_south>" in content:
            return "move", "south"
        elif "<move_east>" in content:
            return "move", "east"
        elif "<move_west>" in content:
            return "move", "west"

        return None, None


async def play_game():
    """Main game loop"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Please set OPENAI_API_KEY in .env file")
        return

    client = OpenAI(api_key=api_key)

    # Create world
    world = World(size=6)

    # Add agents at different starting positions
    world.add_agent("Alice", (0, 0))
    world.add_agent("Bob", (5, 5))
    world.add_agent("Charlie", (2, 3))

    # Create AI controllers
    ais = {
        "Alice": GameAI("Alice", client),
        "Bob": GameAI("Bob", client),
        "Charlie": GameAI("Charlie", client)
    }

    # Add obstacles
    world.add_object(TileType.TREE, (2, 2), 0)
    world.add_object(TileType.TREE, (3, 3), 0)
    world.add_object(TileType.TREE, (1, 4), 0)

    # Add treasures
    world.add_object(TileType.TREASURE, (0, 5), 10)
    world.add_object(TileType.TREASURE, (5, 0), 10)
    world.add_object(TileType.GEM, (2, 4), 5)
    world.add_object(TileType.GEM, (4, 1), 5)
    world.add_object(TileType.COIN, (3, 2), 3)
    world.add_object(TileType.COIN, (1, 1), 3)
    world.add_object(TileType.COIN, (4, 4), 3)

    print("="*60)
    print("🎮 AI AGENT TREASURE HUNT GAME 🎮")
    print("="*60)
    print("\nThree AI agents compete to collect the most treasure!")
    print("Each agent takes turns moving around the 6x6 grid.")
    print("\nStarting game...\n")

    # Show initial world
    print(world.render())
    input("Press Enter to start...")

    # Game loop
    max_turns = 15
    agent_names = ["Alice", "Bob", "Charlie"]

    for turn in range(max_turns):
        world.turn_count = turn + 1
        print(f"\n{'='*60}")
        print(f"TURN {world.turn_count}")
        print(f"{'='*60}")

        for agent_name in agent_names:
            print(f"\n--- {agent_name}'s Turn ---")

            # Get agent's view
            view = world.get_agent_view(agent_name)
            print(view)

            # Agent decides action
            ai = ais[agent_name]
            action, param = ai.decide_action(view, world.render())

            if action == "move":
                success, message = world.move_agent(agent_name, param)
                print(f"\n✓ {message}")
            else:
                print(f"\n✗ No valid action taken")

            # Show updated world
            print(world.render())

            # Small delay for readability
            input(f"Press Enter for next player...")

    # Game over
    print("\n" + "="*60)
    print("🏁 GAME OVER! 🏁")
    print("="*60)
    print(world.render())

    # Determine winner
    winner = max(world.agents.items(), key=lambda x: x[1].score)
    print(f"\n🏆 WINNER: {winner[0]} with {winner[1].score} points! 🏆\n")

    print("Final Inventories:")
    for name, agent in world.agents.items():
        print(f"  {name}: {agent.inventory}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(play_game())
