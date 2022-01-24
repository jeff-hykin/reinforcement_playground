from tools.all_tools import *
from tools.agent_recorder import AgentRecorder

database = AgentRecorder(save_to="resources/datasets.ignore/atari/baselines_pretrained@breakout_custom")

def compress_observations(batch):
    observation_stacks = []
    actions = []
    for index, (raw_image, observation_stack), action in batch:
        observation_stacks.append(observation_stack)
        actions.append(action)
    return to_tensor(observation_stacks), to_tensor(actions)

frame_stacking_size = 4
def compress_raw_images(batch):
    raw_image_stacks = []
    actions = []
    for index, (raw_image, observation_stack), action in batch:
        if index < frame_stacking_size:
            continue
        
        stack = [0]*frame_stacking_size
        for relative_index, each in enumerate(stack):
            offset = (relative_index+1)
            # grab the previous images
            _, ((prev_raw_image, prev_observation_stack), action) = database.load_index(index-offset)
            stack[-offset] = prev_raw_image
        
        raw_image_stacks.append(to_tensor(stack))
        actions.append(action)
    return opencv_image_to_torch_image(to_tensor(raw_image_stacks)), to_tensor(actions)
    
database.create_batch_data(batch_name="balanced64", batch_size=64, preprocessing=compress_observations) 
# database.create_batch_data(batch_name="raw64", batch_size=64, preprocessing=compress_raw_images)

# ~44506 total
# ls -1 resources/datasets.ignore/atari/baselines_pretrained@breakout_custom/balanced64 | wc -l
# rm -rf resources/datasets.ignore/atari/baselines_pretrained@breakout_custom/balanced64;mkdir -p resources/datasets.ignore/atari/baselines_pretrained@breakout_custom/balanced64