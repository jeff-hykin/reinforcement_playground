from agents.informed_vae.wip import * 
from tools.dataset_tools import binary_mnist

__file__ = '/home/jeffhykin/repos/reinforcement_playground/main/agents/informed_vae/mnist_test.py'
# randomize the torch seed
from time import time as now; torch.manual_seed(now())

# 
# train and test
# 
results = []
split = SplitAutoEncoder()
classifier = ImageClassifier()
for each in [9,3,8]:
    train_dataset, test_dataset, train_loader, test_loader = binary_mnist([each])
    
    # reset the task network part (last few layers)
    split.task_network = nn.Sequential(
        nn.Linear(product(split.latent_shape), 2), # binary classification
        nn.Sigmoid(),
    )
    
    # reset the task network part (last few layers)
    classifier.task_network = nn.Sequential(
        nn.Linear(product(classifier.latent_shape), 2), # binary classification
        nn.Sigmoid(),
    )
    
    # create fresh classifier
    fresh_classifier = ImageClassifier()
    
    results.append({
        "split_train": split.fit(loader=train_loader, number_of_epochs=3),
        "split_test": split.test(test_loader),
        "classifier_train": classifier.fit(loader=train_loader, number_of_epochs=3),
        "classifier_test": classifier.test(test_loader),
        "fresh_classifier_train": fresh_classifier.fit(loader=train_loader, number_of_epochs=3),
        "fresh_classifier_test": fresh_classifier.test(test_loader),
    })


import json
from os.path import join, dirname
with open(join(dirname(__file__), 'data.json'), 'w') as outfile:
    json.dump(to_pure(results), outfile)

__file__ = '/home/jeffhykin/repos/reinforcement_playground/main/agents/informed_vae/mnist_test.py'
import json
from os.path import join, dirname
with open(join(dirname(__file__), 'data.json'), 'r') as in_file:
    results = json.load(in_file)
    
for each_digit in results:
    pass
each_digit = results[0]
import silver_spectacle as ss
ss.display("chartjs", {
    "type": 'line',
    "data": {
        "datasets": [
            {
                "label": ['split_train'],
                "data": [ sum(each) for each in each_digit['split_train']],
                "fill": True,
                "tension": 0.1,
                "borderColor": 'rgb(75, 192, 192)',
            },
            {
                "label": 'classifier_train',
                "data": each_digit['classifier_train'],
                "fill": True,
                "tension": 0.1,
                "borderColor": 'rgb(0, 292, 192)',
            },
            {
                "label": 'fresh_classifier_train',
                "data": each_digit['fresh_classifier_train'],
                "fill": True,
                "tension": 0.1,
                "borderColor": 'rgb(0, 92, 192)',
            },
        ]
    },
    "options": {
        "pointRadius": 3,
        "scales": {
            "y": {
                "min": 0,
                "max": 0.5,
            }
        }
    }
})
# 
# importance values
# 
# latent_spaces_for_training  = to_tensor( torch.from_numpy(b.encoder(train_dataset[index][0]).cpu().detach().numpy()) for index in range(len(train_dataset)) if index < 10)
# latent_spaces_for_testing   = to_tensor( torch.from_numpy(b.encoder(test_dataset[index][0]).cpu().detach().numpy()) for index in range(len(test_dataset)) if index < 1)

# import shap

# model = nn.Sequential(b.decoder, nn.Flatten())
# explainer = shap.DeepExplainer(model, latent_spaces_for_training)
# shap_values = explainer.shap_values(latent_spaces_for_testing)


# import numpy
# import functools
# # sum these up elementwise
# summed = numpy.squeeze(functools.reduce(
#     lambda each_new, existing: numpy.add(each_new, existing),
#     # take the absolute value because we just want impactful values, not just neg/pos correlated ones
#     numpy.abs(shap_values),
#     numpy.zeros_like(shap_values[0]),
# ))

# print('summed = ', summed)