import silver_spectacle as ss
from super_map import LazyDict

from tools.all_tools import *
from tools.agent_recorder import AgentRecorder
from prefabs.auto_imitator import AutoImitator

database = AgentRecorder(save_to="resources/datasets.ignore/atari/baselines_pretrained@vectorized_breakout")
auto_imitator = AutoImitator(
    learning_rate=0.001,
    input_shape=(4,84,84),
    latent_shape=(512,),
    output_shape=(4,),
    path="models.ignore/auto_imitator_offline.model",
)

logging = LazyDict(
    should_log=Countdown(size=250),
    correctness_card=ss.DisplayCard("quickLine", []),
    loss_card=ss.DisplayCard("quickLine", []),
)

for index, (observations, actions) in enumerate(database.load_batch_data("64")):
    print(f'batch {index+1}/{database.batch_size}')
    auto_imitator.update_weights(
        batch_of_inputs=observations,
        batch_of_ideal_outputs=actions,
        epoch_index=1,
        batch_index=index
    )
    if logging.should_log():
        logging.correctness_card.send("clear"); logging.correctness_card.send(auto_imitator.logging.proportion_correct_at_index)
        logging.loss_card.send("clear")       ; logging.loss_card.send(auto_imitator.logging.loss_at_index)
    