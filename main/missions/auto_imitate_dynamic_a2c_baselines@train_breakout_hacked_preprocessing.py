import itertools
import torch
import silver_spectacle as ss
from statistics import mean as average
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from tools.all_tools import *
from tools.debug import debug, ic
from tools.all_tools import Countdown, to_tensor, opencv_image_to_torch_image, to_pure
import tools.stat_tools as stat_tools
from tools.agent_recorder import AgentRecorder

from world_builders.atari.baselines_vectorized import Environment
from world_builders.atari.custom_preprocessing import preprocess
from agent_builders.a2c_baselines.main import Agent
from prefabs.baselines_optimizer import RMSpropTFLike
from prefabs.auto_imitator.main import AutoImitator

# 
# Environment
# 
env = preprocess(
    env=gym.make("BreakoutNoFrameskip-v4"),
    frame_buffer_size=4,
    frame_sample_rate=4,
)

# 
# agent
# 
mr_bond = Agent.load("models.ignore/BreakoutNoFrameskip-v4.zip")

#   
# imitator  
#   
auto_imitator = AutoImitator(
    learning_rate=0.00022,
    input_shape=(4,84,84),
    latent_shape=(512,),
    output_shape=(4,),
    path="models.ignore/auto_imitator_long_term_1.model",
)

# 
# training setup
# 
batch_size = 128
mission = LazyDict(
    batch_size=batch_size,
    should_create_batch=Countdown(size=batch_size),
    should_save_model=Countdown(size=256),
    # should_train_imitator=Countdown(size=1), # this will be improved in the future along with randomization
    smoothing_size=256,
    smoother=lambda data: tuple(average(to_pure(each)) for each in bundle(data, bundle_size=mission.smoothing_size)),
    current_batch_details=LazyDict(
        observations=torch.zeros((batch_size, *auto_imitator.input_shape)).float().to(auto_imitator.hardware),
        actions=torch.zeros((batch_size, )).float().to(auto_imitator.hardware),
    ),
    batches=[],
)

logging = LazyDict(
    a2c=LazyDict(
        action_distribution=LazyDict({0:0,1:0,2:0,3:0}),
    ),
    auto_imitator=LazyDict(
        action_distribution=LazyDict({0:0,1:0,2:0,3:0}),
    ),
    should_update_graphs=Countdown(size=1000/2),
    should_print=Countdown(size=100),
    correctness_card=ss.DisplayCard("quickLine", []),
    loss_card=ss.DisplayCard("quickLine", []),
    name_card=ss.DisplayCard("quickMarkdown", f""),
    update_name_card=lambda info: logging.name_card.send(info), 
    update_correctness=lambda data: logging.correctness_card.send("clear").send(tuple(zip(
            # indicies
            range(0, len(data), logging.smoothing_size),
            # values
            mission.smoother(data),
        ))
    ),
    update_loss=lambda data: logging.loss_card.send("clear").send(tuple(zip(
            # indicies
            range(0, len(data), logging.smoothing_size),
            # values
            mission.smoother(data),
        ))
    ),
)

# 
# train and record
# 
def run():
    index = -1
    mr_bond.when_mission_starts()
    for episode_index in itertools.count(0):
        
        mr_bond.observation     = env.reset()
        mr_bond.reward          = 0
        mr_bond.episode_is_over = False
        
        mr_bond.when_episode_starts(episode_index)
        timestep_index = -1
        while not mr_bond.episode_is_over:
            index += 1
            timestep_index += 1
            
            observation_in_torch_form = opencv_image_to_torch_image(observation)
            mr_bond.when_timestep_starts(timestep_index)
            mr_bond.observation, mr_bond.reward, mr_bond.episode_is_over, info = env.step(mr_bond.action)
            # save the data to the current batch
            mission.current_batch_details.observations[index % batch_size] = observation_in_torch_form
            mission.current_batch_details.actions[index % batch_size] = to_pure(mr_bond.action)
            mr_bond.when_timestep_ends(timestep_index)
            
            # 
            # train every single batch (naive approach)
            # 
            if mission.should_create_batch():
                auto_imitator.update_weights(
                    batch_of_inputs=mission.current_batch_details.observations,
                    batch_of_ideal_outputs=mission.current_batch_details.actions,
                    epoch_index=episode_index,
                    batch_index=batch_index,
                )
            # 
            # logging
            # 
            if logging.should_print():
                batch_index = index / batch_size
                print(f'''batch_index:{batch_index}, ''', end="")
                auto_imitator.log()
            
            if logging.should_update_graphs():
                logging.update_name_card()
                logging.update_correctness(auto_imitator.logging.proportion_correct_at_index)
                logging.update_loss(auto_imitator.logging.loss_at_index)
                
        mr_bond.when_episode_ends(episode_index)
    mr_bond.when_mission_ends()
    mr_bond.save()
    env.close()
    
run()