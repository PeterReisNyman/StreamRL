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
        self.current_mode = StreamMode.INTERNAL
        self.message_history: List[Message] = []
        self.current_buffer = ""
        self.token_buffer = TokenBuffer()
        self.incomplete_token_buffer = ""  # Buffer for incomplete tokens with '<'
        self.initial_prompt = None
        self.system_context = ""
        self.has_generated = False

    def split_tokens(self, text: str) -> List[str]:
        """Split text into tokens, keeping tokens with '<' or '>' intact"""
        # Prepend any incomplete token from previous calls
        text = self.incomplete_token_buffer + text
        self.incomplete_token_buffer = ""

        tokens = []
        parts = text.split()

        for i, token in enumerate(parts):
            token = token.strip()
            if not token:
                continue

            # Check if token contains incomplete angle bracket
            if '<' in token and '>' not in token:
                # This token has an opening '<' but no closing '>'
                # Buffer it and wait for more content
                self.incomplete_token_buffer = token
                if i < len(parts) - 1:
                    # If this isn't the last token, add a space
                    self.incomplete_token_buffer += " "
            else:
                tokens.append(token)

        return tokens

    def get_visible_messages(self) -> List[Dict]:
        visible = []
        for msg in self.message_history:
            if msg.visible_to_agent(self.name):
                # Skip empty messages
                if not msg.content or not msg.content.strip():
                    continue

                # Format content: add [Name]: prefix only for OTHER agents
                if msg.agent_name == self.name:
                    content = msg.content
                else:
                    content = f"[{msg.agent_name}]: {msg.content}"

                visible.append({
                    "role": "assistant" if msg.agent_name == self.name else "user",
                    "content": content
                })
        return visible

    def add_visible_token(self, agent_name: str, token: str, mode: StreamMode):
        """Add a single token from another agent"""
        # Filter out empty tokens
        if not token or not token.strip():
            return

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

    def set_initial_prompt(self, prompt: str, system_context: str = ""):
        """Set the initial prompt for this agent"""
        self.initial_prompt = prompt
        self.system_context = system_context

    def generate_next_token(self) -> Optional[tuple[str, StreamMode]]:
        """Generate the next single token with full context from all previous tokens"""
        # First check if we have buffered tokens
        if self.token_buffer.has_more():
            token = self.token_buffer.get_one()
            return token, self.current_mode

        # Build full context with all visible messages
        messages = self.get_visible_messages()

        system_msg = {
            "role": "system",
            "content": f"""You are {self.name}.
{self.system_context}

IMPORTANT: When you respond, just write your message naturally. DO NOT include "[{self.name}]:" or your name in brackets at the start - the system already tracks who is speaking.

You have access to these tools:
- <broadcast>: Switch to broadcast mode - your tokens will be visible to all other agents
- <internal>: Switch to internal mode - your tokens will only be visible to yourself (private thinking)

Current mode: {self.current_mode.value}

You start in INTERNAL mode by default. Other agents cannot see your internal thoughts unless you explicitly use <broadcast> to share with them.
"""
        }

        messages.insert(0, system_msg)

        # Add initial prompt only if we haven't generated anything yet
        if not self.has_generated:
            messages.append({"role": "user", "content": self.initial_prompt})

        # Print what we're sending to OpenAI
        print(f"\n{'='*60}")
        print(f"[{self.name}] SENDING TO OPENAI:")
        print(f"{'='*60}")
        import json
        print(json.dumps(messages, indent=2))
        print(f"{'='*60}\n")

        # Create a NEW stream with updated context
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.7
        )

        # Get first chunk with content
        self.current_buffer = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                self.current_buffer += content

                # Check for tool switches
                if "<broadcast>" in self.current_buffer:
                    parts = self.current_buffer.split("<broadcast>", 1)
                    if parts[0].strip():
                        tokens = self.split_tokens(parts[0].strip())
                        if tokens:
                            self.token_buffer.add_tokens(tokens)
                    self.current_mode = StreamMode.BROADCAST
                    print(f"\n[{self.name}] 🔄 Switched to BROADCAST mode\n")
                    self.current_buffer = parts[1] if len(parts) > 1 else ""
                    self.has_generated = True
                    return self.generate_next_token()

                elif "<internal>" in self.current_buffer:
                    parts = self.current_buffer.split("<internal>", 1)
                    if parts[0].strip():
                        tokens = self.split_tokens(parts[0].strip())
                        if tokens:
                            self.token_buffer.add_tokens(tokens)
                    self.current_mode = StreamMode.INTERNAL
                    print(f"\n[{self.name}] 🔄 Switched to INTERNAL mode\n")
                    self.current_buffer = parts[1] if len(parts) > 1 else ""
                    self.has_generated = True
                    return self.generate_next_token()

                else:
                    # Split content into tokens (by whitespace), keeping '<' intact
                    tokens = self.split_tokens(content)

                    if not tokens:
                        # No tokens, try next chunk
                        continue

                    # Take first token, buffer the rest
                    first_token = tokens[0]
                    remaining_tokens = tokens[1:]

                    if remaining_tokens:
                        self.token_buffer.add_tokens(remaining_tokens)

                    self.has_generated = True
                    return first_token, self.current_mode

        # Stream ended with no tokens
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
        Run agents with token-level interleaving with full context
        prompts: {agent_name: (prompt, system_context)}
        """
        # Set initial prompts for all agents
        for agent_name, (prompt, context) in prompts.items():
            agent = self.agents[agent_name]
            agent.set_initial_prompt(prompt, context)

        agent_names = list(prompts.keys())
        token_counts = {name: 0 for name in agent_names}
        active_agents = set(agent_names)

        print("\n" + "="*60)
        print("Token-Level Interleaved Streaming (True Interleaving)")
        print("="*60 + "\n")

        # Round-robin token generation with full context
        while active_agents:
            for agent_name in agent_names:
                if agent_name not in active_agents:
                    continue

                agent = self.agents[agent_name]
                result = agent.generate_next_token()

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

    def print_agent_perspectives(self):
        """Print what each agent saw during the conversation"""
        print("\n" + "="*60)
        print("WHAT EACH AGENT SAW")
        print("="*60 + "\n")

        for agent_name, agent in self.agents.items():
            print(f"\n{'─'*60}")
            print(f"[{agent_name}]'s Perspective:")
            print(f"{'─'*60}")

            if not agent.message_history:
                print("  (No messages)")
            else:
                for msg in agent.message_history:
                    mode_icon = "📢" if msg.mode == StreamMode.BROADCAST else "🤔"
                    print(f"  {mode_icon} [{msg.agent_name}]: {msg.content}")

            print()

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
    print("Collaborative Problem Solving Demo")
    print("="*60 + "\n")

    # Define the problem
    problem = """Problem: A small town library wants to increase youth engagement (ages 13-18) by 50%
over the next 6 months. They have a limited budget of $5,000. What should they do?"""

    print(problem)
    print("\n" + "="*60 + "\n")

    # Define prompts for each agent
    prompts = {
        "Alice": (
            f"Here's a problem to solve: {problem}\n\nThink creatively about innovative solutions (use <internal> to brainstorm), then use <broadcast> to share your most promising idea.",
            "You are a creative problem solver who thinks outside the box and focuses on innovative, engaging solutions."
        ),
        "Bob": (
            "Observe any proposed solutions. Use <internal> to analyze their logical feasibility, then use <broadcast> to provide analytical feedback or suggest improvements.",
            "You are an analytical thinker who evaluates feasibility, identifies potential issues, and suggests data-driven improvements."
        ),
        "Charlie": (
            "Observe the discussion. Use <internal> to think about practical implementation, then use <broadcast> to provide concrete action steps or identify practical challenges.",
            "You are a practical implementer focused on real-world execution, budgets, timelines, and actionable steps."
        )
    }

    await system.run_interleaved(prompts, max_tokens_per_agent=150)

    # Print what each agent saw
    system.print_agent_perspectives()

    # Print final solution summary
    print("\n" + "="*60)
    print("COLLABORATIVE SOLUTION SUMMARY")
    print("="*60)
    print("\nCombining insights from all three agents:")
    print("- Creative (Alice): Novel ideas and engagement strategies")
    print("- Analytical (Bob): Feasibility analysis and improvements")
    print("- Practical (Charlie): Implementation steps and constraints")
    print("\nSee above for the full collaborative problem-solving process!")
    print("="*60 + "\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
