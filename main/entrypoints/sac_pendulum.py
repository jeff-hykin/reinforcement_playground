from reality_makers.pendulum.main import RealityMaker
from agents.sac.main import Agent
from campaign_managers.simple import start

mr_bond = Agent(
    body_type=RealityMaker.RegularBody
)

start(
    number_of_episodes=100,
    reality=RealityMaker(
        agents=[mr_bond]
    ),
)