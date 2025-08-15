""" Ray management of agents to run multiple agents in parallel
"""

import ray
from patterns.agents import Agent


@ray.remote
class Raygent:
    def __init__(self, agent_config: dict):
        self.agent = Agent(**agent_config)

    def run(self):
        self.agent.run_games()
        return self.agent.replay_data

