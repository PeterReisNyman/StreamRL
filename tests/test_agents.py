from stream_rl.agents import Agent

def test_train_and_generate():
    agent = Agent("a", "")
    agent.train(["hel", "lo"])
    tokens = agent.model.generate()
    assert tokens[:2] == ["hel", "lo"]
