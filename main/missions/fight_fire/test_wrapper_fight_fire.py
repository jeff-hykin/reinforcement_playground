from main.missions.fight_fire.brute_force_fight_fire import get_memory_env, LazyDict
from main.prefabs.general_approximator import GeneralApproximator
from missions.hydra_oracle.a2c_exposed import A2C
from missions.hydra_oracle.policies import ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from blissful_basics import flatten, is_iterable, product
from world_builders.fight_fire.world import World


memory_agent_training_timesteps    = 200
world_shape            = (3, 3)
action_length          = 2
memory_size            = 1
possible_memory_values = [0,1] # each bit is 1 or 0 
observation_size       = product(world_shape) + action_length
input_vector_size      = observation_size + memory_size
verbose                = True


world = World(
    grid_width=world_shape[0],
    grid_height=world_shape[1],
    visualize=False,
    # debug=True,
    fire_locations=[(-1,-1)],
    water_locations=[(0,0)],
)
env = world.Player()


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

wrapped_env = get_memory_env(
    real_env=env,
    memory_shape=(1,),
    RewardPredictor=RewardPredictor,
    PrimaryAgent=PrimaryAgent,
)

model = A2C(MultiInputActorCriticPolicy, wrapped_env, verbose=1)
model.learn(total_timesteps=memory_agent_training_timesteps)

# model = A2C(ActorCriticCnnPolicy, env, verbose=1)
# model.learn(total_timesteps=25_000)


# Questions:
    # how bad is the predictor with a random agent?
    # can the memory agent help the predictor?
    # how bad does A2C do without memory?