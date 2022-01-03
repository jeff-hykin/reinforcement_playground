from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from main.agent_builders.a2c.baselines_recreation.hacked_a2c import A2C
from main.agent_builders.a2c.baselines_optimizer import RMSpropTFLike

from tools.debug import debug, ic

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=16 => 16 environments)
env = make_atari_env(
    'BreakoutNoFrameskip-v4',
    n_envs=16, # from atari optimized hyperparams
    seed=0,
)
# Frame-stacking with 4 frames
env = VecFrameStack(
    env,
    n_stack=4, # from atari optimized hyperparams
)

# 
# train
# 
model = A2C(
    'CnnPolicy', # from atari optimized hyperparams
    env,
    verbose=1,
    ent_coef=0.01, # from atari optimized hyperparams
    vf_coef=0.25, # from atari optimized hyperparams
    policy_kwargs=dict(
        optimizer_class=RMSpropTFLike, # from atari optimized hyperparams
        optimizer_kwargs=dict(eps=1e-5), # from atari optimized hyperparams
    ),
)
model.learn(
    total_timesteps=10_000_000, # from atari optimized hyperparams
) 
model.save("baselines_10_000_000_model.ignore.zip")
  
# 
# test
# 
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()