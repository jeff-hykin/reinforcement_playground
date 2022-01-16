from tools.all_tools import *
from tools.agent_recorder import AgentRecorder

database = AgentRecorder(save_to="resources/datasets.ignore/atari/baselines_pretrained@breakout_custom")

def create_batch_data(self, batch_name, batch_size, preprocessing=lambda each:each):
    # create folder for batch
    batch_path = f"{self.save_to}/{batch_name}"
    FileSystem.ensure_is_folder(batch_path)
    
    # FIXME uniformly random sample from action space
    
    remaining_indicies = set(self.indicies())
    total = math.floor(len(remaining_indicies) / batch_size)
    batch_index = -1
    while len(remaining_indicies) > batch_size:
        batch_index += 1
        entries = random.sample(remaining_indicies, k=batch_size)
        # remove the ones we just sampled
        remaining_indicies = remaining_indicies - set(entries)
        batch = []
        # load all the ones in the batch
        for each_index in entries:
            batch.append(
                large_pickle_load(self.save_to+f"/{each_index}{self._data_file_extension}")
            )
        # do any compression/decompression/augmentation stuff
        batch = preprocessing(batch)
        # save it
        print(f'saving batch {batch_index}/{total}')
        large_pickle_save(batch, f"{batch_path}/{batch_index}")


def compress_observations(batch):
    observation_stacks = []
    actions = []
    for index, (raw_image, observation_stack), action in batch:
        observation_stacks.append(observation_stack)
        actions.append(action)
    return opencv_image_to_torch_image(to_tensor(observation_stacks)), to_tensor(actions)

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
    
# database.create_batch_data(batch_name="preprocessed64", batch_size=64, preprocessing=compress_observations)
database.create_batch_data(batch_name="raw64", batch_size=64, preprocessing=compress_raw_images)
