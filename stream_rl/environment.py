from typing import List

class GuessPhraseEnv:
    """Environment where agents try to reproduce the target phrase."""
    def __init__(self, target: str):
        self.target = target
        self.target_tokens = [target[i:i+3] for i in range(0, len(target), 3)]

    def score(self, tokens: List[str]) -> int:
        correct = 0
        for t1, t2 in zip(tokens, self.target_tokens):
            if t1 == t2:
                correct += 1
        return correct
