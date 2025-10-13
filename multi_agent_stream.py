"""
Multi-Agent Streaming System with Token-Level Interleaving
Each agent streams tokens and can choose between:
- <broadcast>: tokens visible to all agents
- <internal>: tokens only visible to self (private thinking)
"""

from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict, Optional, Iterator
from enum import Enum
from dataclasses import dataclass

load_dotenv()


class StreamMode(Enum):
    BROADCAST = "broadcast"
    INTERNAL = "internal"


@dataclass
class Message:
    agent_name: str
    content: str
    mode: StreamMode

    def visible_to_agent(self, agent_name: str) -> bool:
        if self.mode == StreamMode.BROADCAST:
            return True
        return self.agent_name == agent_name


class TokenBuffer:
    """Buffers tokens (words) and releases them one at a time"""
    def __init__(self):
        self.buffer: List[str] = []

    def add_tokens(self, tokens: List[str]) -> None:
        """Add multiple tokens to buffer"""
        self.buffer.extend(tokens)

    def get_one(self) -> Optional[str]:
        """Get one token from buffer"""
        if self.buffer:
            return self.buffer.pop(0)
        return None

    def has_more(self) -> bool:
        return len(self.buffer) > 0

    def peek_remaining(self) -> List[str]:
        """Show what tokens are buffered"""
        return self.buffer.copy()

    def count(self) -> int:
        """Count buffered tokens"""
        return len(self.buffer)


class Agent:
    def __init__(self, name: str, client: OpenAI, model: str = "gpt-4"):
        self.name = name
        self.client = client
        self.model = model
        self.current_mode = StreamMode.BROADCAST
        self.message_history: List[Message] = []
        self.current_buffer = ""
        self.token_buffer = TokenBuffer()
        self.stream_iterator = None
        self.is_streaming = False
        self.stream_complete = False

    def get_visible_messages(self) -> List[Dict]:
        visible = []
        for msg in self.message_history:
            if msg.visible_to_agent(self.name):
                visible.append({
                    "role": "assistant" if msg.agent_name == self.name else "user",
                    "content": f"[{msg.agent_name}]: {msg.content}"
                })
        return visible

    def add_visible_token(self, agent_name: str, token: str, mode: StreamMode):
        """Add a single token from another agent"""
        # Only add if visible based on mode
        if mode == StreamMode.BROADCAST or agent_name == self.name:
            # Append to last message if from same agent, otherwise create new
            if self.message_history and self.message_history[-1].agent_name == agent_name:
                # Add space before token if not empty
                if self.message_history[-1].content:
                    self.message_history[-1].content += " " + token
                else:
                    self.message_history[-1].content = token
            else:
                self.message_history.append(Message(agent_name, token, mode))

    def start_stream(self, prompt: str, system_context: str = ""):
        """Initialize streaming session"""
        messages = self.get_visible_messages()

        system_msg = {
            "role": "system",
            "content": f"""You are {self.name}.
{system_context}

You have access to these tools:
- <broadcast>: Make your tokens visible to all other agents
- <internal>: Make your tokens private (only you can see them)

Current mode: {self.current_mode.value}
"""
        }

        messages.insert(0, system_msg)
        messages.append({"role": "user", "content": prompt})

        self.stream_iterator = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.7
        )
        self.is_streaming = True
        self.stream_complete = False

    def get_next_token(self) -> Optional[tuple[str, StreamMode]]:
        """Get the next single token to broadcast"""
        # First check if we have buffered tokens
        if self.token_buffer.has_more():
            token = self.token_buffer.get_one()
            return token, self.current_mode

        # Try to get more from stream
        if not self.is_streaming or self.stream_complete:
            return None

        try:
            chunk = next(self.stream_iterator)
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                self.current_buffer += content

                # Check for tool switches
                if "<broadcast>" in self.current_buffer:
                    parts = self.current_buffer.split("<broadcast>", 1)
                    if parts[0].strip():
                        # Tokenize pre-tool content and add to buffer
                        tokens = parts[0].strip().split()
                        self.token_buffer.add_tokens(tokens)
                    self.current_mode = StreamMode.BROADCAST
                    print(f"\n[{self.name}] 🔄 Switched to BROADCAST mode\n")
                    self.current_buffer = parts[1] if len(parts) > 1 else ""
                    return self.get_next_token()  # Recursive call to get actual token

                elif "<internal>" in self.current_buffer:
                    parts = self.current_buffer.split("<internal>", 1)
                    if parts[0].strip():
                        # Tokenize pre-tool content and add to buffer
                        tokens = parts[0].strip().split()
                        self.token_buffer.add_tokens(tokens)
                    self.current_mode = StreamMode.INTERNAL
                    print(f"\n[{self.name}] 🔄 Switched to INTERNAL mode\n")
                    self.current_buffer = parts[1] if len(parts) > 1 else ""
                    return self.get_next_token()

                else:
                    # Split content into tokens (by whitespace)
                    tokens = content.split()

                    if not tokens:
                        # No tokens, try next chunk
                        return self.get_next_token()

                    # Take first token, buffer the rest
                    first_token = tokens[0]
                    remaining_tokens = tokens[1:]

                    if remaining_tokens:
                        self.token_buffer.add_tokens(remaining_tokens)

                    return first_token, self.current_mode
            else:
                return self.get_next_token()  # Try next chunk

        except StopIteration:
            self.stream_complete = True
            self.is_streaming = False
            return None


class MultiAgentSystem:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.agents: Dict[str, Agent] = {}

    def add_agent(self, name: str) -> Agent:
        agent = Agent(name, self.client, self.model)
        self.agents[name] = agent
        print(f"✅ Added agent: {name}")
        return agent

    async def run_interleaved(self, prompts: Dict[str, tuple[str, str]], max_tokens_per_agent: int = 200):
        """
        Run agents with token-level interleaving
        prompts: {agent_name: (prompt, system_context)}
        """
        # Start all streams
        for agent_name, (prompt, context) in prompts.items():
            agent = self.agents[agent_name]
            print(f"\n[{agent_name}] Starting stream...")
            agent.start_stream(prompt, context)

        agent_names = list(prompts.keys())
        token_counts = {name: 0 for name in agent_names}
        active_agents = set(agent_names)

        print("\n" + "="*60)
        print("Token-Level Interleaved Streaming")
        print("="*60 + "\n")

        # Round-robin token generation
        while active_agents:
            for agent_name in agent_names:
                if agent_name not in active_agents:
                    continue

                agent = self.agents[agent_name]
                result = agent.get_next_token()

                if result is None:
                    active_agents.remove(agent_name)
                    print(f"\n[{agent_name}] ✓ Stream complete\n")
                    continue

                token, mode = result
                token_counts[agent_name] += 1

                # Display token with mode indicator
                mode_indicator = "📢" if mode == StreamMode.BROADCAST else "🤔"
                print(f"\n[{agent_name}]{mode_indicator} '{token}'", end="")

                # Show buffered tokens if any
                if agent.token_buffer.has_more():
                    buffered = agent.token_buffer.peek_remaining()
                    buffered_text = " ".join(buffered)
                    print(f"  [EXTRA: {agent.token_buffer.count()} tokens buffered: '{buffered_text}']", end="")

                # Broadcast token to all agents
                for other_agent in self.agents.values():
                    other_agent.add_visible_token(agent_name, token, mode)

                # Check if this agent has reached max tokens
                if token_counts[agent_name] >= max_tokens_per_agent:
                    active_agents.remove(agent_name)
                    print(f"\n[{agent_name}] ⏹ Max tokens reached\n")

        print("\n" + "="*60)
        print("Interleaved streaming complete")
        print("="*60 + "\n")


async def demo():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Please set OPENAI_API_KEY in .env file")
        return

    system = MultiAgentSystem(api_key, model="gpt-4")

    system.add_agent("Alice")
    system.add_agent("Bob")
    system.add_agent("Charlie")

    print("\n" + "="*60)
    print("Multi-Agent Token-Interleaved Demo")
    print("="*60 + "\n")

    # Define prompts for each agent
    prompts = {
        "Alice": (
            "Introduce yourself in 1 short sentence and suggest discussing AI ethics.",
            "You are a creative AI interested in philosophy."
        ),
        "Bob": (
            "Respond briefly to the conversation.",
            "You are a logical AI interested in science."
        ),
        "Charlie": (
            "Add your perspective briefly.",
            "You are a practical AI interested in real-world applications."
        )
    }

    await system.run_interleaved(prompts, max_tokens_per_agent=100)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
