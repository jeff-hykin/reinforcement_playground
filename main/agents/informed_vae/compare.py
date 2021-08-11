#%%
from tools.all_tools import *

from torchvision import datasets, transforms
from tools.basics import product
from tools.pytorch_tools import Network
from tools.record_keeper import RecordKeeper

from tools.basics import *
from tools.ipython_tools import show
from tools.dataset_tools import binary_mnist

from agents.informed_vae.simple_classifier import SimpleClassifier
from agents.informed_vae.split_classifier import SplitClassifier
from agents.informed_vae.classifier_output import ClassifierOutput
#%%
import silver_spectacle as ss
from time import time as now
torch.manual_seed(now())

permute = lambda a_list: sample(a_list, k=len(tuple(a_list)))
if __name__ == "__main__":
    
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger("lightning_logs", name="vae_compare")
    record_keeper = RecordKeeper(test="compare_transfer_v1").sub_record_keeper()
    
    # 
    # perform test on mnist dataset if run directly
    # 
    result_string = ""
    split = SplitClassifier(suppress_output=True, record_keeper=record_keeper); split.name = "split"  ; split.chart  = dict(label="split", backgroundColor='rgb( 0,  92, 192, 0.9)', borderColor='rgb( 0,  92, 192, 0.9)')
    simple = SimpleClassifier(                    record_keeper=record_keeper); simple.name = "simple"; simple.chart = dict(label="simple",backgroundColor='rgb(75, 192, 192, 0.9)', borderColor='rgb(75, 192, 192, 0.9)')
    labels = []
    data = {}
    datasets = []
    for index, each_number in enumerate(permute(range(10))):
        record_keeper.parent_should_include(binary_classification_of=each_number, transfer_index=index)
        # doesn't matter that its binary mnist cause the autoencoder only uses input anyways
        train_dataset, test_dataset, train_loader, test_loader = quick_loader(binary_mnist([each_number]), [5, 1])
        
        fresh = SimpleClassifier(record_keeper=record_keeper, fresh=True); fresh.name = "fresh"; fresh.chart = dict(label="fresh",backgroundColor='rgb(0, 292, 192, 0.9)', borderColor='rgb(0, 292, 192, 0.9)')
        
        models = [split, simple, fresh]
        for model in models:
            model.classifier = ClassifierOutput(input_shape=(30,), output_shape=(2,))
            model.fit(loader=train_loader, max_epochs=1, logger=logger)
            model.number_correct = model.test(loader=test_loader)
        
        result_string += f'{each_number}:\n'+''.join([f'    {each_model.name}: {each_model.number_correct}\n' for each_model in models])
        labels += [each_number]
        for each_model in models:
            data[each_model.name] = data[each_model.name] if each_model.name in data else []
            data[each_model.name].append(each_model.number_correct)
        datasets = [ dict(**each_model.chart, data=data[each_model.name]) for each_model in models ]
        import json
        from os.path import join
        with open('./logs/datasets.dont-sync.json', 'w') as outfile:
            json.dump(dict(labels=labels, datasets=datasets), outfile)
        ss.DisplayCard("chartjs", {
            "type": 'line',
            "options": {
                "pointRadius": 3, # the size of the dots
                "scales": {
                    "y": {
                        "min": 9700,
                        "max": 10000,
                    },
                }
            },
            "data": {
                "labels": labels,
                "datasets": datasets,
            }
        })
        # give intermediate results
        print(result_string)
        record_keeper.save("./logs/records.dont-sync.json")
    
    
    print(result_string)
    # save to file encase connection dies
    FS.write(result_string, to="./log.dont-sync.txt")
    
    # 
    # test inputs/outputs
    # 
    # from tools.basics import *
    # for each_index in range(100):
    #     input_data, correct_output = train_dataset[each_index]
    #     # train_dataset, test_dataset, train_loader, test_loader
    #     guess = [ round(each, ndigits=0) for each in to_pure(model.forward(input_data)) ]
    #     actual = to_pure(correct_output)
    #     index = max_index(guess)
    #     print(f"guess: {guess},\t  index: {index},\t actual: {actual}")
        
    # # 
    # # sample inputs/outputs
    # # 
    # print("showing samples")
    # samples = []
    # for each_index in range(100):
    #     input_data, correct_output = train_dataset[each_index]
    #     output = model.forward(input_data)
    #     sample =  torch.cat(
    #         (
    #             train_dataset.unnormalizer(input_data.to(model.device)),
    #             train_dataset.unnormalizer(output.to(model.device)),
    #         ), 
    #         1
    #     )
    #     show(image_tensor=sample)
    #     samples.append(sample)
# %%
