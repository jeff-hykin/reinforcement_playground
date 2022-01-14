from tools.all_tools import *
from tools.agent_recorder import AgentRecorder

database = AgentRecorder(save_to="resources/datasets.ignore/atari/baselines_pretrained@vectorized_breakout")

def compress(batch):
    observations, actions = zip(*batch)
    return opencv_image_to_torch_image(to_tensor(observations)), to_tensor(actions)
    
database.create_batch_data(batch_name="tensor64", batch_size=64, preprocessing=compress)
