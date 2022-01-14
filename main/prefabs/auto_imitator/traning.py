import silver_spectacle as ss
from super_map import LazyDict

from tools.all_tools import *
from tools.agent_recorder import AgentRecorder
from prefabs.auto_imitator import AutoImitator

database = AgentRecorder(save_to="resources/datasets.ignore/atari/baselines_pretrained@vectorized_breakout")
def learning_rate(timestep_index):
    return 0.000197 - (timestep_index * 0.0001 * 0.00004)

auto_imitator = AutoImitator(
    learning_rate=learning_rate,
    input_shape=(4,84,84),
    latent_shape=(512,),
    output_shape=(4,),
    path="models.ignore/auto_imitator_offline_9.model",
)

logging = LazyDict(
    smoothing_size=128,
    should_log=Countdown(size=1000),
    should_print=Countdown(size=100),
    correctness_card=ss.DisplayCard("quickLine", []),
    loss_card=ss.DisplayCard("quickLine", []),
    update_correctness=lambda data: logging.correctness_card.send("clear").send(tuple(((index+1)*logging.smoothing_size, each) for index, each in enumerate(tuple(average(to_pure(each)) for each in bundle(data, bundle_size=logging.smoothing_size))))),
    update_loss=lambda        data: logging.loss_card.send(       "clear").send(tuple(((index+1)*logging.smoothing_size, each) for index, each in enumerate(tuple(average(to_pure(each)) for each in bundle(data, bundle_size=logging.smoothing_size))))),
)

for index, (observations, actions) in enumerate(database.load_batch_data("64")):
    if logging.should_print(): print(f'batch {index+1}/{database.batch_size}')
    auto_imitator.update_weights(
        batch_of_inputs=opencv_image_to_torch_image(observations),
        batch_of_ideal_outputs=actions,
        epoch_index=1,
        batch_index=index
    )
    
    if logging.should_log() or index == 0:
        logging.update_correctness(auto_imitator.logging.proportion_correct_at_index)
        logging.update_loss(auto_imitator.logging.loss_at_index)
        auto_imitator.save()
    