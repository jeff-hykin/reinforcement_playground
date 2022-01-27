import silver_spectacle as ss
from super_map import LazyDict, Map
from super_hash import super_hash

from tools.all_tools import *
from tools.basics import to_pure, Countdown
from tools.debug import debug
from tools.file_system_tools import FileSystem
from tools.stat_tools import bundle, proportionalize
from tools.agent_recorder import AgentRecorder
from tools.schedulers import create_linear_rate
from tools.percent import Percent
from informative_iterator import ProgressBar


from prefabs.fail_fast_check import is_significantly_below_other_curves
from prefabs.auto_imitator.main import AutoImitator
from prefabs.helpful_fitness_measures import trend_up, average
from prefabs.auto_imitator.preprocess_dataset import compress_observations, compress_raw_images


# best so far, starts with learning_rate of 0.00022752556564934162, gets 0.559375
# 0.00009873062729, 0.5520833333333334
# 0.000275410365795725, 0.5477294921875


logging = LazyDict(
    smoothing_size=64,
    should_update_graphs=Countdown(size=1000/2),
    should_print=Countdown(size=100),
    correctness_card=ss.DisplayCard("quickLine", []),
    loss_card=ss.DisplayCard("quickLine", []),
    name_card=ss.DisplayCard("quickMarkdown", f""),
    smoother=lambda data: tuple(average(to_pure(each)) for each in bundle(data, bundle_size=logging.smoothing_size)),
    update_name_card=lambda info: logging.name_card.send(info), 
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
# testing
# 
testing_batch_generator = database.load_batch_data("balanced64", epochs=1, batches_per_epoch=300)
# create one huge batch
print("creating test batch")
test_observations = []
test_actions = []
for progress, (observations, a2c_actions) in ProgressBar(testing_batch_generator, seconds_per_print=0.2):
    test_observations += [ each for each in observations ]
    test_actions      += [ each for each in a2c_actions ]
test_observations = to_tensor(test_observations).to(device)
test_actions = to_tensor(test_actions).to(device)
print("creating hashes")
test_observation_hashes = tuple(super_hash(each.tolist()) for _, each in ProgressBar(test_observations))
def test(model):
    return model.correctness(
        model.forward(test_observations),
        test_actions
    )
print("test_batch created")

# 
# training
# 
training_batch_generator = database.load_batch_data("balanced64", epochs=math.inf, batches_per_epoch=6000)
other_curves = []
def train(
        base_learning_rate=0.00022,
        learning_rate_shrink=0.1,
        number_of_epochs=200,
    ):
    path = f"models.ignore/auto_imitator_hacked_compressed_preprocessing_ManyEpochs_{base_learning_rate:.12f}.model"
    FileSystem.delete(path)
    
    
    iterations = number_of_epochs * training_batch_generator.batches_per_epoch
    auto_imitator = AutoImitator(
        learning_rate=create_linear_rate(
            base_learning_rate=base_learning_rate,
            min_learning_rate=base_learning_rate * (1 - learning_rate_shrink),
            number_of_training_steps=iterations,
        ),
        input_shape=(4,84,84),
        latent_shape=(512,),
        output_shape=(4,),
        path=path,
    )
    training_log = Map(
        this_score_curve=[0],
    )
    
    # 
    # training
    # 
    for progress, (observations, actions) in ProgressBar(training_batch_generator, iterations=iterations, seconds_per_print=30, disable_logging=False):
        if super_hash(observations[0].tolist()) in test_observation_hashes:
            continue
        
        auto_imitator.update_weights(
            batch_of_inputs=observations,
            batch_of_ideal_outputs=actions,
            epoch_index=training_batch_generator.epoch_index,
            batch_index=training_batch_generator.batch_index,
        )
        
        if progress.updated:
            accuracy = Percent(training_log.this_score_curve[-1]*100)
            print(f"learning_rate: {auto_imitator.learning_rate_scheduler.current_value:.12f}, train_accuracy: {str(accuracy)}, test_accuracy: {test(auto_imitator):.3f}, trial: {len(other_curves)}, epoch:{training_batch_generator.epoch_index}", end="")
        
        if logging.should_update_graphs() or progress.index == 0:
            logging.update_name_card(f"""
                ### trial: {len(other_curves)}, epoch: {training_batch_generator.epoch_index}, learning_rate: {auto_imitator.learning_rate_scheduler.current_value:.12f}
            """.replace("                ", ""))
            logging.update_correctness(auto_imitator.logging.proportion_correct_at_index)
            logging.update_loss(auto_imitator.logging.loss_at_index)
            # fail fast check
            training_log.this_score_curve = logging.smoother(auto_imitator.logging.proportion_correct_at_index)
            if is_significantly_below_other_curves(training_log.this_score_curve, other_curves):
                print("ending early due to being below other curves")
                return training_log.this_score_curve
            # hard coded early stopping
            if progress.index >= 3000:
                if training_log.this_score_curve[-1] < 0.4:
                    print("ending early from poor performance")
                    return training_log.this_score_curve
            
            auto_imitator.save()
    
    # 
    # testing
    # 
    for progress, (observations, actions) in ProgressBar(testing_batch_generator, iterations=iterations, seconds_per_print=0.5, disable_logging=False):
        pass
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
            base_learning_rate=trial.suggest_loguniform('learning_rate', 0.00007, 0.00030),
            number_of_epochs=trial.suggest_categorical('number_of_epochs', [ 100, ]),
            # trial.suggest_categorical
            # trial.suggest_discrete_uniform
            # trial.suggest_float
            # trial.suggest_int
            # trial.suggest_loguniform
            # trial.suggest_uniform
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