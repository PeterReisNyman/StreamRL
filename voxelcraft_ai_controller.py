"""
VoxelCraft AI Agent Controller
Connects AI agents to VoxelCraft world with token-level interleaving
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
import json
import time
from typing import List, Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import threading
import random
from multi_agent_stream import StreamMode

load_dotenv()


class VoxelAction(Enum):
    """Actions an AI agent can take in VoxelCraft"""
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    LOOK_UP = "look_up"
    LOOK_DOWN = "look_down"
    PLACE_BLOCK = "place_block"
    BREAK_BLOCK = "break_block"
    WAIT = "wait"


class BlockType(Enum):
    """Block types in VoxelCraft"""
    GRASS = 1
    DIRT = 2
    STONE = 3
    SAND = 4
    WATER = 5
    WOOD = 6
    LEAVES = 7
    SNOW = 8


@dataclass
class VoxelPosition:
    x: float
    y: float
    z: float
    yaw: float  # rotation left/right (radians)
    pitch: float  # rotation up/down (radians)


class VoxelCraftClient:
    """Client to interact with VoxelCraft server"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", room: str = "alpha"):
        self.base_url = base_url
        self.room = room
        self.client_id = None
        self.position = VoxelPosition(0, 120, 0, 0, 0)

    def generate_client_id(self, name: str) -> str:
        """Generate a unique client ID"""
        import uuid
        self.client_id = f"{name}_{uuid.uuid4().hex[:8]}"
        return self.client_id

    def send_position(self, name: str, color: str = "#ff0000"):
        """Send position update to server"""
        if not self.client_id:
            self.generate_client_id(name)

        payload = {
            "type": "pos",
            "clientId": self.client_id,
            "room": self.room,
            "name": name,
            "x": round(self.position.x, 2),
            "y": round(self.position.y, 2),
            "z": round(self.position.z, 2),
            "yaw": round(self.position.yaw, 3),
            "pitch": round(self.position.pitch, 3),
            "color": color
        }

        try:
            requests.post(f"{self.base_url}/publish", json=payload, timeout=1)
        except Exception as e:
            print(f"Error sending position: {e}")

    def place_block(self, block_x: int, block_y: int, block_z: int, block_type: BlockType):
        """Place a block in the world"""
        key = f"{block_x},{block_y},{block_z}"
        payload = {
            "type": "edit",
            "room": self.room,
            "key": key,
            "id": block_type.value
        }

        try:
            requests.post(f"{self.base_url}/publish", json=payload, timeout=1)
            print(f"Placed {block_type.name} at ({block_x}, {block_y}, {block_z})")
        except Exception as e:
            print(f"Error placing block: {e}")

    def break_block(self, block_x: int, block_y: int, block_z: int):
        """Break a block in the world"""
        key = f"{block_x},{block_y},{block_z}"
        payload = {
            "type": "edit",
            "room": self.room,
            "key": key,
            "id": 0  # 0 means remove block
        }

        try:
            requests.post(f"{self.base_url}/publish", json=payload, timeout=1)
            print(f"Broke block at ({block_x}, {block_y}, {block_z})")
        except Exception as e:
            print(f"Error breaking block: {e}")

    def move(self, dx: float = 0, dy: float = 0, dz: float = 0):
        """Move relative to current position"""
        self.position.x += dx
        self.position.y += dy
        self.position.z += dz

    def rotate(self, dyaw: float = 0, dpitch: float = 0):
        """Rotate camera"""
        self.position.yaw += dyaw
        self.position.pitch += dpitch
        # Clamp pitch
        import math
        self.position.pitch = max(-math.pi/2, min(math.pi/2, self.position.pitch))

    def get_snapshot(self) -> Dict:
        """Get current world state"""
        try:
            response = requests.get(f"{self.base_url}/snapshot?room={self.room}", timeout=2)
            return response.json()
        except Exception as e:
            print(f"Error getting snapshot: {e}")
            return {}


class VoxelAgent:
    """AI agent that operates in VoxelCraft world"""

    def __init__(self, name: str, client: OpenAI, voxel_client: VoxelCraftClient,
                 color: str = "#ff0000", model: str = "gpt-4"):
        self.name = name
        self.client = client
        self.voxel_client = voxel_client
        self.color = color
        self.model = model
        self.action_history: List[str] = []
        self.current_mode = StreamMode.INTERNAL
        self.message_history: List[Dict] = []
        self.task = None

    def set_task(self, task: str):
        """Set the agent's current task/goal"""
        self.task = task

    def get_world_description(self) -> str:
        """Get a text description of the world state"""
        pos = self.voxel_client.position
        snapshot = self.voxel_client.get_snapshot()

        desc = f"""
=== {self.name}'s View ===
Position: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})
Rotation: Yaw={pos.yaw:.2f}, Pitch={pos.pitch:.2f}

World Info:
- Seed: {snapshot.get('seed', 'unknown')}
- Edits: {len(snapshot.get('edits', {}))} blocks placed
- Other players: {len(snapshot.get('clients', {}))}

Other Players:
"""
        clients = snapshot.get('clients', {})
        for cid, cdata in clients.items():
            if cdata.get('name') != self.name:
                desc += f"  - {cdata.get('name')}: ({cdata.get('x'):.1f}, {cdata.get('y'):.1f}, {cdata.get('z'):.1f})\n"

        return desc

    def get_visible_messages(self) -> List[Dict]:
        """Get messages this agent can see"""
        visible = []
        for msg in self.message_history:
            mode = msg.get('mode', StreamMode.BROADCAST)
            agent_name = msg.get('agent', self.name)

            # Can see broadcast or own messages
            if mode == StreamMode.BROADCAST or agent_name == self.name:
                if agent_name == self.name:
                    visible.append({
                        "role": "assistant",
                        "content": msg['content']
                    })
                else:
                    visible.append({
                        "role": "user",
                        "content": f"[{agent_name}]: {msg['content']}"
                    })
        return visible

    def decide_action(self) -> Tuple[Optional[VoxelAction], Optional[Dict]]:
        """AI decides what action to take"""
        world_desc = self.get_world_description()
        messages = self.get_visible_messages()

        system_prompt = f"""You are {self.name}, an AI agent in a 3D voxel world (like Minecraft).

Your current task: {self.task or "Explore and build"}

You have access to these tools:
- <broadcast>: Share your thoughts with other agents
- <internal>: Think privately
- <move_forward>: Move forward in the direction you're facing
- <move_backward>: Move backward
- <move_left>: Strafe left
- <move_right>: Strafe right
- <move_up>: Move up (fly)
- <move_down>: Move down
- <turn_left>: Turn 45° left
- <turn_right>: Turn 45° right
- <look_up>: Look up
- <look_down>: Look down
- <place_grass>: Place grass block ahead
- <place_stone>: Place stone block ahead
- <place_wood>: Place wood block ahead
- <break_block>: Break block ahead
- <wait>: Wait/observe

Choose ONE action. Start in <internal> mode to think, then optionally use <broadcast> to communicate."""

        messages.insert(0, {"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": world_desc})

        print(f"\n[{self.name}] Thinking...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )

            content = response.choices[0].message.content
            print(f"[{self.name}] Response: {content[:100]}...")

            # Parse action from content
            action, params = self._parse_action(content)

            # Store message
            self.message_history.append({
                'agent': self.name,
                'content': content,
                'mode': self.current_mode
            })

            return action, params

        except Exception as e:
            print(f"[{self.name}] Error: {e}")
            return VoxelAction.WAIT, {}

    def _parse_action(self, content: str) -> Tuple[Optional[VoxelAction], Dict]:
        """Parse action from AI response"""
        content_lower = content.lower()

        # Check mode switches
        if "<broadcast>" in content:
            self.current_mode = StreamMode.BROADCAST
            print(f"[{self.name}] 🔄 Switched to BROADCAST")
        elif "<internal>" in content:
            self.current_mode = StreamMode.INTERNAL
            print(f"[{self.name}] 🔄 Switched to INTERNAL")

        # Parse movement actions
        if "<move_forward>" in content_lower:
            return VoxelAction.MOVE_FORWARD, {}
        elif "<move_backward>" in content_lower:
            return VoxelAction.MOVE_BACKWARD, {}
        elif "<move_left>" in content_lower:
            return VoxelAction.MOVE_LEFT, {}
        elif "<move_right>" in content_lower:
            return VoxelAction.MOVE_RIGHT, {}
        elif "<move_up>" in content_lower:
            return VoxelAction.MOVE_UP, {}
        elif "<move_down>" in content_lower:
            return VoxelAction.MOVE_DOWN, {}
        elif "<turn_left>" in content_lower:
            return VoxelAction.TURN_LEFT, {}
        elif "<turn_right>" in content_lower:
            return VoxelAction.TURN_RIGHT, {}
        elif "<look_up>" in content_lower:
            return VoxelAction.LOOK_UP, {}
        elif "<look_down>" in content_lower:
            return VoxelAction.LOOK_DOWN, {}

        # Parse building actions
        elif "<place_grass>" in content_lower:
            return VoxelAction.PLACE_BLOCK, {"block_type": BlockType.GRASS}
        elif "<place_stone>" in content_lower:
            return VoxelAction.PLACE_BLOCK, {"block_type": BlockType.STONE}
        elif "<place_wood>" in content_lower:
            return VoxelAction.PLACE_BLOCK, {"block_type": BlockType.WOOD}
        elif "<break_block>" in content_lower:
            return VoxelAction.BREAK_BLOCK, {}
        elif "<wait>" in content_lower:
            return VoxelAction.WAIT, {}

        return VoxelAction.WAIT, {}

    def execute_action(self, action: VoxelAction, params: Dict):
        """Execute the chosen action"""
        import math

        if action == VoxelAction.MOVE_FORWARD:
            # Move in the direction we're facing
            dx = -math.sin(self.voxel_client.position.yaw) * 2
            dz = -math.cos(self.voxel_client.position.yaw) * 2
            self.voxel_client.move(dx=dx, dz=dz)

        elif action == VoxelAction.MOVE_BACKWARD:
            dx = math.sin(self.voxel_client.position.yaw) * 2
            dz = math.cos(self.voxel_client.position.yaw) * 2
            self.voxel_client.move(dx=dx, dz=dz)

        elif action == VoxelAction.MOVE_LEFT:
            dx = -math.cos(self.voxel_client.position.yaw) * 2
            dz = math.sin(self.voxel_client.position.yaw) * 2
            self.voxel_client.move(dx=dx, dz=dz)

        elif action == VoxelAction.MOVE_RIGHT:
            dx = math.cos(self.voxel_client.position.yaw) * 2
            dz = -math.sin(self.voxel_client.position.yaw) * 2
            self.voxel_client.move(dx=dx, dz=dz)

        elif action == VoxelAction.MOVE_UP:
            self.voxel_client.move(dy=2)

        elif action == VoxelAction.MOVE_DOWN:
            self.voxel_client.move(dy=-2)

        elif action == VoxelAction.TURN_LEFT:
            self.voxel_client.rotate(dyaw=math.pi/4)

        elif action == VoxelAction.TURN_RIGHT:
            self.voxel_client.rotate(dyaw=-math.pi/4)

        elif action == VoxelAction.LOOK_UP:
            self.voxel_client.rotate(dpitch=-0.3)

        elif action == VoxelAction.LOOK_DOWN:
            self.voxel_client.rotate(dpitch=0.3)

        elif action == VoxelAction.PLACE_BLOCK:
            # Place block in front of agent
            block_type = params.get('block_type', BlockType.GRASS)
            pos = self.voxel_client.position
            # Calculate block position ahead
            bx = int(pos.x - math.sin(pos.yaw) * 3)
            by = int(pos.y)
            bz = int(pos.z - math.cos(pos.yaw) * 3)
            self.voxel_client.place_block(bx, by, bz, block_type)

        elif action == VoxelAction.BREAK_BLOCK:
            # Break block in front
            pos = self.voxel_client.position
            bx = int(pos.x - math.sin(pos.yaw) * 3)
            by = int(pos.y)
            bz = int(pos.z - math.cos(pos.yaw) * 3)
            self.voxel_client.break_block(bx, by, bz)

        # Update position on server
        self.voxel_client.send_position(self.name, self.color)
        print(f"[{self.name}] ✓ Executed {action.name}")


async def run_voxelcraft_agents():
    """Run multiple AI agents in VoxelCraft world"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Please set OPENAI_API_KEY in .env file")
        return

    client = OpenAI(api_key=api_key)

    # Create VoxelCraft clients for each agent
    room = "ai_agents"

    alice_client = VoxelCraftClient(room=room)
    alice_client.position = VoxelPosition(0, 120, 0, 0, 0)

    bob_client = VoxelCraftClient(room=room)
    bob_client.position = VoxelPosition(10, 120, 0, 1.57, 0)

    charlie_client = VoxelCraftClient(room=room)
    charlie_client.position = VoxelPosition(0, 120, 10, -1.57, 0)

    # Create AI agents
    alice = VoxelAgent("Alice", client, alice_client, color="#ff0000")
    alice.set_task("Build a tower out of stone blocks")

    bob = VoxelAgent("Bob", client, bob_client, color="#00ff00")
    bob.set_task("Build a house out of wood blocks")

    charlie = VoxelAgent("Charlie", client, charlie_client, color="#0000ff")
    charlie.set_task("Create a garden with grass blocks")

    agents = [alice, bob, charlie]

    print("="*60)
    print("🎮 AI AGENTS IN VOXELCRAFT 🎮")
    print("="*60)
    print("\nThree AI agents are now operating in VoxelCraft!")
    print("\nTo view them, open:")
    print(f"  http://127.0.0.1:8000/index.html?room={room}&name=Observer")
    print("\n" + "="*60)

    # Send initial positions
    for agent in agents:
        agent.voxel_client.send_position(agent.name, agent.color)

    print("\nStarting AI agent loop (20 turns)...\n")

    # Main loop: agents take turns
    max_turns = 20
    for turn in range(max_turns):
        print(f"\n{'='*60}")
        print(f"TURN {turn + 1}/{max_turns}")
        print(f"{'='*60}")

        for agent in agents:
            print(f"\n--- {agent.name}'s Turn ---")

            # Agent decides action
            action, params = agent.decide_action()

            if action:
                # Execute action
                agent.execute_action(action, params)

                # Share message with other agents if in broadcast mode
                if agent.current_mode == StreamMode.BROADCAST and agent.message_history:
                    last_msg = agent.message_history[-1]
                    for other in agents:
                        if other != agent:
                            other.message_history.append(last_msg)

            # Small delay
            time.sleep(0.5)

        # Pause between turns
        time.sleep(1)

    print("\n" + "="*60)
    print("🏁 AI AGENTS COMPLETE! 🏁")
    print("="*60)
    print("\nCheck VoxelCraft to see what they built!")
    print(f"  http://127.0.0.1:8000/index.html?room={room}&name=Observer")


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_voxelcraft_agents())
