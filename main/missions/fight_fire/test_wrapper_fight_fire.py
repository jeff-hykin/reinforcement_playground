from copy import deepcopy

import silver_spectacle as ss
from blissful_basics import flatten, is_iterable, product, print, countdown
from trivial_torch_tools.generics import to_pure

from main.missions.fight_fire.brute_force_fight_fire import get_memory_env, get_transformed_env, LazyDict
from main.prefabs.general_approximator import GeneralApproximator
from main.tools.basics import numbers_to_accumulated_numbers
from main.tools.universe.runtimes import basic as basic_runtime
from main.tools.universe.agent import Skeleton, Enhancement, enhance_with
from main.agent_builders.universal_random import Agent as RandomAgent
from missions.hydra_oracle.a2c_exposed import A2C
from missions.hydra_oracle.policies import ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from world_builders.fight_fire.world import World


world_shape            = (3, 1)
action_length          = 2
memory_size            = 1
possible_memory_values = [0,1] # each bit is 1 or 0 
observation_size       = product(world_shape) + action_length
input_vector_size      = observation_size + memory_size
verbose                = True


world = World(
    grid_width=world_shape[0],
    grid_height=world_shape[1],
    # visualize=True,
    # debug=True,
    # fast_as_possible=True,
    fire_locations=[(-1,-1)],
    water_locations=[(0,0)],
)
real_env = world.Player()

# 
# Customize Agents to Env
# 
if True:
    class CustomRandomAgent(RandomAgent):
        def when_timestep_starts(self):
            action = self.reaction_space.sample()
            # make it binary
            for index in range(len(action)):
                if action[index] > 0.5:
                    action[index] = 1.0
                else:
                    action[index] = 0.0
            self.timestep.reaction = action
    
# 
# Customize Reward Predictor
# 
class RewardPredictor:
    latest = None
    def __init__(self, input_space):
        RewardPredictor.latest = self
        self.approximator = GeneralApproximator(
            input_shape=(-1,), # this approximator doesn't need/use input_shape
            output_shape=(1,),
            # max_number_of_points=2000,
        )
        self.losses = []
        self.major_losses = []
        self.table = LazyDict()
    
    def predict(self, *args):
        output = flatten(self.approximator.predict(*args))[0]
        return output
    
    def update(self, inputs, correct_outputs):
        *data, action_bit_1, action_bit_2, memory_bit = flatten(to_pure(inputs))
        if len(inputs) == 1:
            output = self.predict(inputs)
            difference = abs(output - correct_outputs[0])
            self.losses.append(difference)
            if difference > 0.0001: # NOTE: Hardcoded to a specific env
                self.major_losses.append(LazyDict(
                    difference=difference,
                    input=inputs[0],
                    output=output,
                    correct_output=correct_outputs[0],
                ))
            
        return self.approximator.fit(inputs, correct_outputs)

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
    
    # 
    # create trajectory
    # 
    timesteps_for_evaluation = 200
    log_rate = 10
    with print.indent.block("Creating Trajectory", disable=True):
        import torch
        from copy import deepcopy
        from tools.universe.timestep import Timestep
        trajectory = []
        done = True
        for index in range(timesteps_for_evaluation):
            if done:
                observation = real_env.reset()
            
            reaction = real_env.action_space.sample()
            # force to be binary because action-space is just a hack for A2C to work
            for index in range(len(reaction)):
                if reaction[index] > 0.5:
                    reaction[index] = 1.0
                else:
                    reaction[index] = 0.0
            
            next_observation, reward, done, info = real_env.step(reaction)
            trajectory.append(Timestep(
                index=deepcopy(index),
                observation=deepcopy(observation),
                reaction=deepcopy(reaction),
                reward=deepcopy(reward),
                is_last_step=deepcopy(done),
                hidden_info=deepcopy(info),
            ))
            observation = next_observation
    
    trajectory = tuple(trajectory)
            
    a2c_memory_actions_rewards    = []
    random_memory_actions_rewards = []
    perfect_memory_agent_rewards  = []
    multi_plot = ss.DisplayCard("multiLine", 
        dict(
            random=random_memory_actions_rewards,
            perfect=perfect_memory_agent_rewards,
            a2c=a2c_memory_actions_rewards,
        ),
        dict(
            vertical_label="Mem Reward per timestep",
            horizonal_label="timesteps",
        ),
    )
    loss_data = LazyDict(
        random=[],
        perfect=[],
        a2c=[],
    )
    
    
    # 
    # how bad is the predictor with trained A2C
    #
    with print.indent.block("### A2C"):
        reward_total = 0
        model = None
        # TODO: change the reward predictor, where every prediction involves re-running the A2C model through the known trajectory (generating new memory values), then use those as the new inputs/outputs
        class ExpensivePredictor:
            
            def __init__(self, *args, **kwargs):
                self.approximator = GeneralApproximator(
                    input_shape=(-1,), # this approximator doesn't need/use input_shape
                    output_shape=(1,),
                )
                
            def predict(self, inputs):
                return flatten(self.approximator.predict(inputs))[0]
                
            def update(self, inputs, correct_outputs):
                # 
                # create new correct values
                # 
                recursive_runtime = create_runtime(
                    agent=LazyDict(
                        choose_action=lambda state: model.predict(state)[0],
                    ),
                    env=get_memory_env(
                        real_env=real_env,
                        real_trajectory=trajectory,
                        memory_shape=(1,),
                        RewardPredictor=RewardPredictor,
                        PrimaryAgent=random_agent_factory,
                    ),
                    max_timestep_index=timesteps_for_evaluation,
                )
                inputs = []
                correct_outputs = []
                for episode_index, timestep in recursive_runtime:
                    prev_memory_bit, real_observation, primary_agent_action = timestep.observation.values()
                    memory_action = timestep.reaction
                    
                    inputs.append(
                        (real_observation, primary_agent_action, memory_action)
                    )
                    correct_outputs.append(
                        timestep.hidden_info.real_reward
                    )
                    
                self.approximator = GeneralApproximator( input_shape=(-1,), output_shape=(1,),)
                # update with entirely new values
                return self.approximator.fit(inputs, correct_outputs)
        
        memory_env = get_memory_env(
            real_env=real_env,
            real_trajectory=trajectory,
            memory_shape=(1,),
            RewardPredictor=ExpensivePredictor,
            PrimaryAgent=random_agent_factory,
        )
        model = A2C(MultiInputActorCriticPolicy, memory_env, verbose=1)
        class A2cAgent(Skeleton):
            def when_timestep_starts(self):
                self.timestep.reaction = model.predict(self.timestep.observation)[0]
        model.learn(total_timesteps=timesteps_for_evaluation)
        
        # reset the reward predictor for evaluation
        memory_env = get_memory_env(
            real_env=real_env,
            real_trajectory=trajectory,
            memory_shape=(1,),
            RewardPredictor=ExpensivePredictor,
            PrimaryAgent=random_agent_factory,
        )
        runtime = basic_runtime(
            agent=A2cAgent(
                reaction_space=memory_env.action_space,
                observation_space=memory_env.observation_space,
            ),
            env=memory_env,
            max_timestep_index=timesteps_for_evaluation,
        )
        number_of_timesteps = 0
        should_log = countdown(size=log_rate)
        reward_per_timestep_over_time = []
        with print.indent:
            for trajectory_timestep_index, (episode_index, timestep) in enumerate(runtime):
                reward_total += timestep.reward
                number_of_timesteps += 1
                if should_log():
                    # print(f'''timestep = {timestep}''')
                    reward_per_timestep = reward_total/number_of_timesteps
                    reward_per_timestep_over_time.append(reward_per_timestep)
                    multi_plot.send(dict(
                        a2c=[
                            [ trajectory_timestep_index, reward_per_timestep ],
                        ]  
                    ))
                    print(f'''number_of_timesteps = {number_of_timesteps}''')
                    print(f'''reward_total = {reward_total}''')
                    print(f'''number_of_episodes = {episode_index + 1}''')
                    print(f'''reward_total/number_of_episodes = {(reward_total/(episode_index + 1))}''')
                    print(f'''reward_per_timestep = {(reward_per_timestep)}''')
        print(f'''reward_per_timestep_over_time = {reward_per_timestep_over_time}''')
        a2c_memory_actions_rewards = list(reward_per_timestep_over_time)
    loss_data.a2c = deepcopy(numbers_to_accumulated_numbers(RewardPredictor.latest.losses))
    
    # 
    # how does the perfect agent do?
    # 
    with print.indent.block("### Perfect agent"):
        reward_total = 0
        memory_env = get_memory_env(
            real_env=real_env,
            real_trajectory=trajectory,
            memory_shape=(1,),
            RewardPredictor=RewardPredictor,
            PrimaryAgent=random_agent_factory,
        )
        
        class PerfectAgent(Skeleton):
            def when_timestep_starts(self):
                self.timestep.reaction = self.choose_action(self.timestep.observation)
            
            def choose_action(self, state):
                import numpy
                prev_memory, observation, primary_agent_action = state.values()
                # preserve a positive memory value
                if flatten(prev_memory)[0] == 1:
                    output = numpy.array([1]) 
                    return numpy.array([1]) 
                else:
                    position_layer = observation[0]
                    top_row = position_layer[0]
                    left_cell = top_row[0]
                    memory_value = 1 if left_cell else 0
                    output = numpy.array([memory_value]) 
                    return output
            
        runtime = basic_runtime(
            agent=PerfectAgent(
                reaction_space=memory_env.action_space,
                observation_space=memory_env.observation_space,
            ),
            env=memory_env,
            max_timestep_index=timesteps_for_evaluation,
        )
        number_of_timesteps = 0
        should_log = countdown(size=log_rate)
        reward_per_timestep_over_time = []
        with print.indent:
            for trajectory_timestep_index, (episode_index, timestep) in enumerate(runtime):
                reward_total += timestep.reward
                number_of_timesteps += 1
                if should_log():
                    print(f'''timestep = {timestep}''')
                    reward_per_timestep = reward_total/number_of_timesteps
                    reward_per_timestep_over_time.append(reward_per_timestep)
                    multi_plot.send(dict(
                        perfect=[
                            [ trajectory_timestep_index, reward_per_timestep ],
                        ]  
                    ))
                    print(f'''number_of_timesteps = {number_of_timesteps}''')
                    print(f'''reward_total = {reward_total}''')
                    print(f'''number_of_episodes = {episode_index + 1}''')
                    print(f'''reward_total/number_of_episodes = {(reward_total/(episode_index + 1))}''')
                    print(f'''reward_per_timestep = {(reward_per_timestep)}''')
        print(f'''reward_per_timestep_over_time = {reward_per_timestep_over_time}''')
        perfect_memory_agent_rewards = list(reward_per_timestep_over_time)
    loss_data.perfect = deepcopy(numbers_to_accumulated_numbers(RewardPredictor.latest.losses))
    
    # 
    # how bad is the predictor with a random agent?
    # 
    with print.indent.block("### Random Agent"):
        reward_total = 0
        memory_env = get_memory_env(
            real_env=real_env,
            real_trajectory=trajectory,
            memory_shape=(1,),
            RewardPredictor=RewardPredictor,
            PrimaryAgent=random_agent_factory,
        )
        runtime = basic_runtime(
            agent=CustomRandomAgent(
                reaction_space=memory_env.action_space,
                observation_space=memory_env.observation_space,
            ),
            env=memory_env,
            max_timestep_index=timesteps_for_evaluation,
        )
        number_of_timesteps = 0
        should_log = countdown(size=log_rate)
        reward_per_timestep_over_time = []
        with print.indent.block(disable=True):
            for trajectory_timestep_index, (episode_index, timestep) in enumerate(runtime):
                reward_total += timestep.reward
                number_of_timesteps += 1
                if should_log():
                    # print(f'''timestep = {timestep}''')
                    reward_per_timestep = reward_total/number_of_timesteps
                    reward_per_timestep_over_time.append(reward_per_timestep)
                    multi_plot.send(dict(
                        random=[
                            [ trajectory_timestep_index, reward_per_timestep ],
                        ]  
                    ))
                    # print(f'''number_of_timesteps = {number_of_timesteps}''')
                    # print(f'''reward_total = {reward_total}''')
                    # print(f'''number_of_episodes = {episode_index + 1}''')
                    # print(f'''reward_total/number_of_episodes = {(reward_total/(episode_index + 1))}''')
                    # print(f'''reward_per_timestep = {(reward_per_timestep)}''')
        print(f'''reward_per_timestep_over_time = {reward_per_timestep_over_time}''')
        random_memory_actions_rewards = list(reward_per_timestep_over_time)
    loss_data.random = deepcopy(numbers_to_accumulated_numbers(RewardPredictor.latest.losses))
    
    multi_plot = ss.DisplayCard("multiLine", 
        dict(
            random=random_memory_actions_rewards,
            perfect=perfect_memory_agent_rewards,
            a2c=a2c_memory_actions_rewards,
        ),
        dict(
            title="Reward Over Time",
            vertical_label="Mem Reward per timestep",
            horizonal_label=f"{timesteps_for_evaluation} timesteps",
        ),
    )
    from tools.stat_tools import bundle, average
    
    loss_plots = ss.DisplayCard("multiLine",
        {
            # apply some smoothing
            each_key : tuple(average(to_pure(each_bundle)) for each_bundle in bundle(each_values, bundle_size=log_rate))
                for each_key, each_values in loss_data.items()
        },
        dict(
            title="Prediction Gap",
            vertical_label="Accumulated Loss",
            horizonal_label="timesteps",
        ),
    )
    from main.tools.afrl_tools import save_all_charts_to
    save_all_charts_to("./main/missions/fight_fire/test_get_memory_env.html")