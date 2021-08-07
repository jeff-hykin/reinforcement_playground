#%%
from tools.all_tools import *

from torchvision import datasets, transforms
from tools.basics import product
from tools.pytorch_tools import Network

from tools.basics import *
from tools.ipython_tools import show
from tools.dataset_tools import binary_mnist

from agents.informed_vae.simple_classifier import SimpleClassifier
from agents.informed_vae.split_classifier import SplitClassifier
from agents.informed_vae.classifier_output import ClassifierOutput
#%%

if __name__ == "__main__":
    
    # 
    # perform test on mnist dataset if run directly
    # 
    result_string = ""
    split = SplitClassifier(suppress_output=True); split.name = "split"
    simple = SimpleClassifier(); simple.name = "simple"
    for each in [9,8,3,5,0,7,1]:
        # doesn't matter that its binary mnist cause the autoencoder only uses input anyways
        train_dataset, test_dataset, train_loader, test_loader = quick_loader(binary_mnist([each]), [5, 1])
        
        fresh = SimpleClassifier(); fresh.name = "fresh"
        
        models = [split, simple, fresh]
        for model in models:
            model.classifier = ClassifierOutput(input_shape=(30,), output_shape=(2,))
            model.fit(loader=train_loader, max_epochs=3)
            model.number_correct = model.test(loader=test_loader)
        
        result_string += f'{each}:\n'+''.join([f'    {each.name}: {each.number_correct}\n' for each in models])
        # give intermediate results
        print(result_string)
    
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
