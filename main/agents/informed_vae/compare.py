from tools.all_tools import *

from torchvision import datasets, transforms
from tools.basics import product
from tools.pytorch_tools import Network

from agents.informed_vae.simple_classifier import SimpleClassifier
from agents.informed_vae.split_classifier import SplitClassifier

if __name__ == "__main__":
    from tools.basics import *
    from tools.ipython_tools import show
    from tools.dataset_tools import binary_mnist
    
    # 
    # perform test on mnist dataset if run directly
    # 
    split = SplitClassifier(); split.name = "split"
    simple = SimpleClassifier(); simple.name = "simple"
    for each in [9,8,3]:
        # doesn't matter that its binary mnist cause the autoencoder only uses input anyways
        train_dataset, test_dataset, train_loader, test_loader = quick_loader(binary_mnist([each]), [5, 1])
        
        fresh = SimpleClassifier(); fresh.name = "fresh"
        
        models = [split, simple, fresh]
        for model in models:
            model.fit(loader=train_loader, number_of_epochs=3)
            model.number_correct = model.test(loader=test_loader)
        
        print('each = ', each)
        for each in models:
            print(f'    {each.name}: {each.number_correct}')
    
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