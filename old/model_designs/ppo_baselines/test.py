import gym

from stable_baselines3 import PPO
from old.environments.atari.main import Environment
import silver_spectacle as ss

# 
# setup the model
# 
env = Environment()
model = PPO("MlpPolicy", env, verbose=1)
model = model.load("./1200000_breakout_ppo.ignore.zip")

# 
# test & record
# 
results = []
obs = env.reset()
for timestep in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    results.append([timestep, reward])
    # env.render()
    if done:
        obs = env.reset()

env.close()

# 
# display
# 
ss.DisplayCard("quickScatter", results)