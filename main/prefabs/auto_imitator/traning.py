import silver_spectacle as ss
from super_map import LazyDict

from tools.all_tools import *
from tools.agent_recorder import AgentRecorder
from prefabs.auto_imitator import AutoImitator

from prefabs.helpful_fitness_measures import trend_up, average

database = AgentRecorder(save_to="resources/datasets.ignore/atari/baselines_pretrained@vectorized_breakout")
# best so far, starts with learning_rate of 0.00022752556564934162, gets 0.559375
# 0.00009873062729, 0.5520833333333334
# 0.000275410365795725, 0.5477294921875

logging = LazyDict(
    smoothing_size=128,
    should_log=Countdown(size=1000),
    should_print=Countdown(size=100),
    correctness_card=ss.DisplayCard("quickLine", []),
    loss_card=ss.DisplayCard("quickLine", []),
    name_card=ss.DisplayCard("quickMarkdown", f""),
    update_name_card=lambda info: name_card.send(info), 
    update_correctness=lambda data: logging.correctness_card.send("clear").send(tuple(((index+1)*logging.smoothing_size, each) for index, each in enumerate(tuple(average(to_pure(each)) for each in bundle(data, bundle_size=logging.smoothing_size))))),
    update_loss=lambda        data: logging.loss_card.send(       "clear").send(tuple(((index+1)*logging.smoothing_size, each) for index, each in enumerate(tuple(average(to_pure(each)) for each in bundle(data, bundle_size=logging.smoothing_size))))),
)
    
# 
# training
# 
training_number = 0
def train(base_learning_rate):
    global training_number
    training_number += 1
    
    def learning_rate(timestep_index):
        # reduce by orders of magnitude over time
        min_rate = base_learning_rate/(10 * 1)
        flexible_part = base_learning_rate - min_rate
        return min_rate + ((database.size-timestep_index)/database.size * flexible_part)
    
    auto_imitator = AutoImitator(
        learning_rate=learning_rate,
        input_shape=(4,84,84),
        latent_shape=(512,),
        output_shape=(4,),
        path=f"models.ignore/auto_imitator_hypertuning_2_{base_learning_rate}.model",
    )

    for index, (observations, actions) in enumerate(database.load_batch_data("64")):
        if logging.should_print(): print(f'trial: {training_number}, learning_rate: {learning_rate(index)}, batch {index+1}/{database.batch_size}')
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
    
    smoothed_correctness = tuple(average(to_pure(each)) for each in bundle(auto_imitator.logging.proportion_correct_at_index, bundle_size=logging.smoothing_size))
    print(f'training_number = {training_number}, max stable correctness: {max(smoothed_correctness)}')
    return smoothed_correctness


# 
# 
# tuner
# 
#
def tune_hyperparams(number_of_trials, fitness_func):
    import optuna
    # connect the trial-object to hyperparams and setup a measurement of fitness
    objective_func = lambda trial: fitness_func(
        train(
            base_learning_rate=trial.suggest_loguniform('learning_rate', 0.00009, 0.0003),
        )
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=number_of_trials)
    return study

# 
# 
# run
# 
# 
study = tune_hyperparams(number_of_trials=100, fitness_func=max)