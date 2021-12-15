import gym

from stable_baselines3 import PPO
from old.environments.atari.main import Environment
import silver_spectacle as ss
from main.tools.basics import large_pickle_load, large_pickle_save, project_folder

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
observation = env.reset()
for timestep in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    large_pickle_save(observation, project_folder+"/old/model_designs/ppo_baselines/autoencoding_dataset/observations/"+timestep+".pickle")
    large_pickle_save(action, project_folder+"/old/model_designs/ppo_baselines/autoencoding_dataset/actions/"+timestep+".pickle")
    observation, reward, done, info = env.step(action)
    results.append([timestep, reward])
    if done:
        observation = env.reset()

env.close()

# 
# display
# 
ss.DisplayCard("quickScatter", results)