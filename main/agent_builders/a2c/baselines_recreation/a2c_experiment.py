from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from main.agent_builders.a2c.baselines_recreation.hacked_a2c import A2C

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=16, seed=0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

print('env.observation_space = ', env.observation_space)

model = A2C('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=25_000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()