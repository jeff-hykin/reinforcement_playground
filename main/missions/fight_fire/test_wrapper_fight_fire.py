from main.missions.fight_fire.brute_force_fight_fire import get_memory_env, get_transformed_env, LazyDict
from main.prefabs.general_approximator import GeneralApproximator
from missions.hydra_oracle.a2c_exposed import A2C
from missions.hydra_oracle.policies import ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from blissful_basics import flatten, is_iterable, product, print, countdown
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
real_env = world.Player()


# 
# Random agent
# 
def random_action_maker(action_space):
    def random_action(*args, **kwargs):
        action = action_space.sample()
        # print(f'''action = {action}''')
        return action
    return random_action
def random_agent_factory(observation_space, action_space):
    return LazyDict(
        # choose_action= lambda state: action_space.sample(),
        choose_action=random_action_maker(action_space),
    )

# 
# reward predictor
# 
def RewardPredictor(*args, **kwargs):
    approximator = GeneralApproximator(
        input_shape=(-1,), # this approximator doesn't need/use input_shape
        output_shape=(1,),
    )
    return LazyDict(
        predict=lambda *args: flatten(approximator.predict(*args))[0],
        update=lambda inputs, correct_outputs: approximator.fit(inputs, correct_outputs),
    )

# 
# runtime
# 
import itertools
import math
from copy import deepcopy
from tools.universe.timestep import Timestep

def create_runtime(agent, env, max_timestep_index=math.inf, max_episode_index=math.inf):
    agent_data = LazyDict()
    timestep_index = -1
    for episode_index in itertools.count(0): # starting at 0
        agent_data.previous_timestep = Timestep(index=-2,)
        agent_data.timestep = Timestep(index=-1)
        agent_data.next_timestep = Timestep(
            index=0,
            observation=deepcopy(env.reset()),
            is_last_step=False,
        )
        while not agent_data.timestep.is_last_step:
            timestep_index += 1
            if max_timestep_index < timestep_index:
                break
            
            agent_data.previous_timestep = agent_data.timestep
            agent_data.timestep          = agent_data.next_timestep
            agent_data.next_timestep     = Timestep(index=agent_data.next_timestep.index+1)
            
            action = agent.choose_action(agent_data.timestep.observation)
            
            observation, reward, is_last_step, agent_data.timestep.hidden_info = env.step(action)
            agent_data.next_timestep.observation = deepcopy(observation)
            agent_data.timestep.reward           = deepcopy(reward)
            agent_data.timestep.is_last_step     = deepcopy(is_last_step)
            
            yield episode_index, agent_data.timestep
        
        if max_timestep_index < timestep_index:
            break
            

# 
# 
# visual tool
# 
# 
def multi_plot(data, vertical_label=None, horizonal_label=None, title=None, color_key={}):
    import silver_spectacle as ss
    datasets = []
    labels = {}
    for each_key, each_line in data.items():
        values = []
        for x, y in enumerate(each_line):
            labels[x] = None
            values.append(y)
        
        datasets.append(dict(
            label=each_key,
            data=values,
            fill=False,
            tension=0.4,
            cubicInterpolationMode='monotone',
            backgroundColor=color_key.get(each_key, 'rgb(0, 292, 192, 0.5)'),
            borderColor=color_key.get(each_key, 'rgb(0, 292, 192, 0.5)'),
            color=color_key.get(each_key, 'rgb(0, 292, 192, 0.5)'),
        ))
        
    
    labels = list(labels.keys())
    return ss.DisplayCard("chartjs", {
        "type": 'line',
        "data": {
            "labels": labels,
            "datasets": datasets,
        },
        "options": {
            "plugins": {
                "title": {
                    "display": (not (not title)),
                    "text": title,
                }
            },
            "pointRadius": 3, # the size of the dots
            "scales": {
                "x": {
                    "title": {
                        "display": horizonal_label,
                        "text": horizonal_label,
                    },
                },
                "y": {
                    "title": {
                        "display": vertical_label,
                        "text": vertical_label,
                    },
                    # "min": 50,
                    # "max": 100,
                },
            }
        }
    })


# 
# 
# tests
# 
# 
if True:
    # Questions:
        # how bad is the predictor with a random agent?
        # can the memory agent help the predictor?
        # how bad does A2C do without memory?
    
    transformed_env = get_transformed_env(
        real_env=real_env,
        memory_shape=(1,),
        memory_agent_factory=random_agent_factory,
    )

    memory_env = get_memory_env(
        real_env=real_env,
        memory_shape=(1,),
        RewardPredictor=RewardPredictor,
        PrimaryAgent=random_agent_factory,
    )
    
    timesteps_for_evaluation = 10_000

    # 
    # how bad is the predictor with trained A2C
    #
    with print.indent:
        reward_total = 0
        env = memory_env
        model = A2C(MultiInputActorCriticPolicy, memory_env, verbose=1)
        model.learn(total_timesteps=timesteps_for_evaluation)
        def choose_action(state):
            return model.predict(state)[0]
        runtime = create_runtime(
            agent=LazyDict(
                choose_action=choose_action,
            ),
            env=env,
            max_timestep_index=timesteps_for_evaluation,
        )
        number_of_timesteps = 0
        should_log = countdown(size=200)
        reward_per_timestep_over_time = []
        with print.indent:
            for episode_index, timestep in runtime:
                reward_total += timestep.reward
                number_of_timesteps += 1
                if should_log():
                    reward_per_timestep_over_time.append(reward_total/number_of_timesteps)
                    print(f'''number_of_timesteps = {number_of_timesteps}''')
                    print(f'''reward_total = {reward_total}''')
                    print(f'''number_of_episodes = {episode_index + 1}''')
                    print(f'''reward_total/number_of_episodes = {(reward_total/(episode_index + 1))}''')
                    print(f'''reward_total/number_of_timesteps = {(reward_total/number_of_timesteps)}''')
        a2c_memory_actions_rewards = list(reward_per_timestep_over_time)
    
    # 
    # how bad is the predictor with a random agent?
    # 
    with print.indent:
        reward_total = 0
        env = memory_env
        runtime = create_runtime(
            agent=LazyDict(
                choose_action=random_action_maker(env.action_space),
            ),
            env=env,
            max_timestep_index=timesteps_for_evaluation,
        )
        number_of_timesteps = 0
        should_log = countdown(size=200)
        reward_per_timestep_over_time = []
        with print.indent:
            for episode_index, timestep in runtime:
                reward_total += timestep.reward
                number_of_timesteps += 1
                if should_log():
                    reward_per_timestep_over_time.append(reward_total/number_of_timesteps)
                    print(f'''number_of_timesteps = {number_of_timesteps}''')
                    print(f'''reward_total = {reward_total}''')
                    print(f'''number_of_episodes = {episode_index + 1}''')
                    print(f'''reward_total/number_of_episodes = {(reward_total/(episode_index + 1))}''')
                    print(f'''reward_total/number_of_timesteps = {(reward_total/number_of_timesteps)}''')
        random_memory_actions_rewards = list(reward_per_timestep_over_time)
    
    multi_plot(
        vertical_label="Mem Reward per timestep",
        horizonal_label="200 timesteps",
        data=dict(
            random_memory_actions=reward_per_timestep_over_time,
            a2c_memory_actions_rewards=a2c_memory_actions_rewards,
        ),
    )
                
    # model = A2C(ActorCriticCnnPolicy, env, verbose=1)
    # model.learn(total_timesteps=25_000)

    
    
    
    