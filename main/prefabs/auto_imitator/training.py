import silver_spectacle as ss
from super_map import LazyDict

from tools.basics import to_pure, Countdown
from tools.stat_tools import bundle
from tools.agent_recorder import AgentRecorder

from prefabs.fail_fast_check import is_significantly_below_other_curves
from prefabs.auto_imitator.main import AutoImitator
from prefabs.helpful_fitness_measures import trend_up, average
from prefabs.auto_imitator.preprocess_dataset import compress_observations, compress_raw_images


# best so far, starts with learning_rate of 0.00022752556564934162, gets 0.559375
# 0.00009873062729, 0.5520833333333334
# 0.000275410365795725, 0.5477294921875


logging = LazyDict(
    smoothing_size=128,
    smoother=lambda data: tuple(average(to_pure(each)) for each in bundle(data, bundle_size=logging.smoothing_size)),
    should_log=Countdown(size=1000),
    should_print=Countdown(size=100),
    correctness_card=ss.DisplayCard("quickLine", []),
    loss_card=ss.DisplayCard("quickLine", []),
    name_card=ss.DisplayCard("quickMarkdown", f""),
    update_name_card=lambda info: name_card.send(info), 
    update_correctness=lambda data: logging.correctness_card.send("clear").send(tuple(zip(
            # indicies
            range(0, len(data), logging.smoothing_size),
            # values
            logging.smoother(data),
        ))
    ),
    update_loss=lambda data: logging.loss_card.send("clear").send(tuple(zip(
            # indicies
            range(0, len(data), logging.smoothing_size),
            # values
            logging.smoother(data),
        ))
    ),
)

database = AgentRecorder(
    save_to="resources/datasets.ignore/atari/baselines_pretrained@breakout_custom"
)

# 
# training
# 
other_curves = []
def train(base_learning_rate):
    def learning_rate(timestep_index):
        # reduce by orders of magnitude over time
        min_rate = base_learning_rate/(10 * 1)
        flexible_part = base_learning_rate - min_rate
        return min_rate + ((database.size-timestep_index)/database.number_of_batches * flexible_part)
    
    auto_imitator = AutoImitator(
        learning_rate=learning_rate,
        input_shape=(4,84,84),
        latent_shape=(512,),
        output_shape=(4,),
        path=f"models.ignore/auto_imitator_hacked_compressed_preprocessing_2_{base_learning_rate}.model",
    )
    
    batch_size = 64
    for index, (observations, actions) in enumerate(database.load_batch_data("preprocessed64")):
        if logging.should_print(): print(f'trial: {len(other_curves)+1}, learning_rate: {auto_imitator.learning_rate_scheduler.current_value}, batch {index+1}/{database.number_of_batches}')
        auto_imitator.update_weights(
            batch_of_inputs=observations,
            batch_of_ideal_outputs=actions,
            epoch_index=1,
            batch_index=index
        )
        
        if logging.should_log() or index == 0:
            logging.update_name_card(f"trial: {len(other_curves)+1}, learning_rate: {auto_imitator.learning_rate_scheduler.current_value}")
            logging.update_correctness(auto_imitator.logging.proportion_correct_at_index)
            logging.update_loss(auto_imitator.logging.loss_at_index)
            # fail fast check
            this_score_curve = logging.smoother(auto_imitator.logging.proportion_correct_at_index)
            if is_significantly_below_other_curves(this_score_curve, other_curves):
                print("ending early due to being below other curves")
                return other_curves
            auto_imitator.save()
    
    smoothed_correctness = logging.smoother(auto_imitator.logging.proportion_correct_at_index)
    print('smoothed_correctness = ', smoothed_correctness)
    other_curves.append(smoothed_correctness)
    print(f'training_number = {len(other_curves)+1}, max stable correctness: {max(smoothed_correctness)}')
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
            base_learning_rate=trial.suggest_loguniform('learning_rate', 0.00007, 0.00035),
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