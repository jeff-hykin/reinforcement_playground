from environments.atari.main import Environment
from agents.dqn.main import Agent
from runtimes.simple import run

# help(Environment)
enduro_env = Environment(game='Enduro')
agent = Agent(action_space=enduro_env.action_space)
run(
    number_of_episodes=100,
    env=env,
    agent=agent,
)