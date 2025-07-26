import ray
from mcts.agent import Agent


@ray.remote
class AgentWorker:
    def __init__(self, agent_config: dict):
        self.agent = Agent(**agent_config)

    def run(self):
        self.agent.run_games()
        return self.agent.replay_data
