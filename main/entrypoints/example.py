from environments.unity.main import Environment
from agents.basic_sac.main import Agent
from runtimes.simple import run

env = Environment()
agent = Agent(action_space=env.action_space)
run(
    number_of_episodes=100,
    env=env,
    agent=agent,
)