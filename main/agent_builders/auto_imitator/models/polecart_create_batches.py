from tools.all_tools import *
from tools.agent_recorder import AgentRecorder

database = AgentRecorder(
    path="resources/datasets.ignore/gym_basics/a2c_mine@cartpole"
)

def compress_observations(batch):
    observations = []
    actions = []
    for index, observation, action in batch:
        observations.append(observation)
        actions.append(action)
    return to_tensor(observations), to_tensor(actions)

database.create_batch_data(batch_name="512", batch_size=512, preprocessing=compress_observations) 