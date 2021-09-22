from environments.pendulum.main import Environment
from agents.sac.main import Agent
from runtimes.simple import run

env = Environment()
mr_bond = Agent(action_space=env.action_space)
run(
    number_of_episodes=100,
    env=env,
    agent=mr_bond,
)