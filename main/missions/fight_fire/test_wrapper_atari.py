from main.missions.fight_fire.brute_force_fight_fire import wrap, LazyDict
from main.prefabs.general_approximator import GeneralApproximator
from missions.hydra_oracle.a2c_exposed import A2C
from missions.hydra_oracle.policies import ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from blissful_basics import flatten, is_iterable
from world_builders.atari.world import World

world = World(
    game="pong",
    mode=None,
    difficulty=None,
    obs_type="image",
    frameskip=4,
    visualize=False,
    debug=False,
    random_seed=None,
)
env = world.Player(
    repeat_action_probability=0.0,
    full_action_space=False,
)

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)
from stable_baselines3.common.env_util import make_atari_env
# env = make_atari_env('PongNoFrameskip-v4', n_envs=4, seed=0)
# env = VecFrameStack(env, n_stack=4) # Frame-stacking with 4 frames

# Random agent
def PrimaryAgent(observation_space, action_space):
    return LazyDict(
        choose_action= lambda state: action_space.sample(),
    )

def RewardPredictor(*args, **kwargs):
    approximator = GeneralApproximator(
        input_shape=(-1,), # this approximator doesn't need/use input_shape
        output_shape=(1,),
    )
    return LazyDict(
        predict=lambda *args: flatten(approximator.predict(*args))[0],
        update=lambda inputs, correct_outputs: approximator.fit(inputs, correct_outputs),
    )

wrapped_env = wrap(
    real_env=env,
    memory_shape=(1,),
    RewardPredictor=RewardPredictor,
    PrimaryAgent=PrimaryAgent,
)

model = A2C(MultiInputActorCriticPolicy, wrapped_env, verbose=1)
model.learn(total_timesteps=2500)

# model = A2C(ActorCriticCnnPolicy, env, verbose=1)
# model.learn(total_timesteps=25_000)

