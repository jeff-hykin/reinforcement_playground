#%%

from torchvision import datasets, transforms
from tools.basics import product
from tools.pytorch_tools import Network
from tools.record_keeper import RecordKeeper
import silver_spectacle as ss

from tools.all_tools import *
from tools.basics import *
from tools.ipython_tools import show
from tools.dataset_tools import binary_mnist
# from tools.record_keeper import ExperimentCollection

from agents.informed_vae.simple_classifier import SimpleClassifier
from agents.informed_vae.split_classifier import SplitClassifier
from agents.informed_vae.classifier_output import ClassifierOutput
#%%

# setup the experiment
collection = ExperimentCollection("vae_comparison")
number_of_runs_for_redundancy = 5
for each_greater_iteration in range(number_of_runs_for_redundancy):
    # new_experiment auto-increments the experiment number within a collection
    with collection.new_experiment(
            test="binary_mnist",
            seed=now(), # randomize each run
            binary_class_order=list(range(10)), # fixed order, but has been randomized in the past
            train_test_ratio=[5, 1],
        ) as record_keeper:
        
        # set random seed
        torch.manual_seed(record_keeper.seed)
        
        # 
        # transfer learning iterations
        # 
        placeholder = record_keeper.sub_record_keeper()
        split  = SplitClassifier (record_keeper=placeholder)
        simple = SimpleClassifier(record_keeper=placeholder)
        for index, each_number in enumerate(record_keeper.binary_class_order):
            
            # load dataset
            train_dataset, test_dataset, train_loader, test_loader = quick_loader(binary_mnist([each_number]), record_keeper.train_test_ratio)
            
            # record: 1. how much training has been done 2. what class the iteration was for 3. misc 
            iteration_record_keeper = record_keeper.sub_record_keeper(
                binary_class=each_number,
                transfer_learning_iteration=0,
            )
            
            # connect record keepers to models
            fresh = SimpleClassifier(record_keeper=iteration_record_keeper, fresh=True)
            split.record_keeper.parent.parent  = iteration_record_keeper # model.record_keeper.parent        has the model parameters/hyper-parameters
            simple.record_keeper.parent.parent = iteration_record_keeper # model.record_keeper.parent.parent is the placeholder we set above, or a previous iteration_record_keeper
            
            # 
            # train & test
            # 
            models = [split, simple, fresh]
            for model in models:
                # each model has a classifier layer, this resets it
                model.classifier = ClassifierOutput(input_shape=(30,), output_shape=(2,))
                # training (data sent to record keeper)
                model.fit(loader=train_loader, max_epochs=1)
                # testing (data sent to record keeper)
                model.test(loader=test_loader)
            
            
# %%
