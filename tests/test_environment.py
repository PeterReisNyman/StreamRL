from stream_rl.environment import GuessPhraseEnv

def test_score():
    env = GuessPhraseEnv("hello")
    tokens = ["hel", "lo"]
    assert env.score(tokens) == 2
    tokens = ["hel", "la"]
    assert env.score(tokens) == 1
