import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from world_builders.atari.baselines_vectorized import Environment
from prefabs.baselines_optimizer import RMSpropTFLike
from statistics import mean as average

# Parallel environments
env = Environment(
    name='BreakoutNoFrameskip-v4',
    n_envs=1, # from atari optimized hyperparams
    n_stack=4, # 4 frames, from atari optimized hyperparams
    seed=0,
)

# 
# load
# 
model = A2C.load("models.ignore/BreakoutNoFrameskip-v4.zip")

episode_rewards = [[]]
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    episode_rewards[-1] += rewards.tolist()
    # start a new episode
    if dones[0]:
        average_episode_reward = average(tuple(sum(each) for each in episode_rewards))
        print('average_episode_reward = ', average_episode_reward)
        episode_rewards.append([])
    