from environments.pendulum.main import RealityMaker
from agents.sac.main import Agent
from runtimes.simple import run

mr_bond = Agent(
    body_type=RealityMaker.RegularBody
)

run(
    number_of_episodes=100,
    env=RealityMaker(mr_bond),
    agent=mr_bond,
)