import silver_spectacle as ss
from tools.all_tools import *
from tools.record_keeper import ExperimentCollection
from tools.liquid_data import LiquidData
from super_map import LazyDict, Map
from statistics import mean as average

collection = ExperimentCollection(FS.local_path("vae_comparison"))
# collection.records[0] example:
# {
#     "testing": true,
#     "average_loss": 7.984413504600525e-05,
#     "accuracy": 0.9729,
#     "correct": 9729,
#     "input_shape": [
#         1,
#         28,
#         28
#     ],
#     "mid_shape": [
#         30
#     ],
#     "output_shape": [
#         2
#     ],
#     "learning_rate": 0.01,
#     "momentum": 0.5,
#     "model": "SimpleClassifier",
#     "fresh": true,
#     "binary_class": 2,
#     "transfer_learning_iteration": 0,
#     "test": "binary_mnist",
#     "seed": 1630425327.0230868,
#     "binary_class_order": [
#         0,
#         1,
#         2
#     ],
#     "train_test_ratio": [
#         5,
#         1
#     ],
#     "experiment_number": 2,
#     "error_number": 0,
#     "had_error": false,
#     "experiment_start_time": 1630425327.0230994,
#     "experiment_end_time": 1630425422.6926785,
#     "experiment_duration": 95.66957902908325
# }

# 
# basic filter + map
# 
data = LiquidData(collection.records).only_keep_if(lambda each:
        # no runs with errors allowed
        not each["had_error"]
        # no training (or other) values
        and each["testing"]
        # only for binary_mnist
        and each["test"] == "binary_mnist_not_normalized"
        # and each["test"] == "binary_mnist"
        # make sure binary_class_order didnt change
        #and each["binary_class_order"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # make sure decoder importance didnt change
        #and not (each["model"] == "split" and each["decoder_importance"] != 0.1)
        # make sure latent_shape is the same
        and (
                list(each["latent_shape"]or[]) == [30]
            # (for simple_classifier its called mid_shape)
            or list(each["mid_shape"   ]or[]) == [30]
        )
    ).map(lambda each: {
        **each,
        # change the model name for Simple+Fresh
        "model": each["model"] if not each["fresh"] else "FreshClassifier",
    })

# 
# aggregate into data we want to display
# 
bin_class_agg = data.bundle_by(
        "model",
    # nested bundle
    ).bundle_by(
        "binary_class",
    # convert lists-of-dictionaries into a dictionary-of-lists 
    # one dictionary per binary class
    ).aggregate(
    # average within each model, within binary_class, across all experiments
    )

def agg1(records_within_class):
    return {
        
        # for each of these, their list elements are all the same so just grab the first
        "model": records_within_class["model"][0],
        "binary_class": records_within_class["binary_class"][0],
        "transfer_learning_iteration": records_within_class["transfer_learning_iteration"][0],
        
        # for these however, we want to average across experiments
        "correct": average(records_within_class["correct"]),
        "accuracy": average(records_within_class["accuracy"]),
        "average_loss": average(records_within_class["average_loss"]),
    }

averaged = bin_class_agg.map(agg1)
averaged.bundles[0][0]
    

# 
# convert to chart-friendly format
# 
model_color_map = {
    "FreshClassifier": "rgb(0, 292, 192, 0.9)",
    "SimpleClassifier": "rgb(75, 192, 192, 0.9)",
    "SplitImportanceClassifier": 'rgb(100, 92, 192, 0.9)',
    "SplitClassifier": 'rgb(0, 92, 192, 0.9)',
    "SplitRootClassifier": 'rgb(200, 92, 192, 0.9)',
}

chart_data = averaged.aggregate()
# remove the redundant data (e.g. model name)
def agg2(each):
    try:
        return {
            # again, they're all the same so just grab the first
            "model": each["model"][0],
            # these are not all the same, keep them as a list (x-axis values)
            "binary_class": each["binary_class"],
            # these are not all the same, keep them as a list (y-axis values)
            "correct": each["correct"],
            # these are not all the same, keep them as a list (alterative y-axis values)
            "accuracy": each["accuracy"],
            # these are not all the same, keep them as a list (alterative y-axis values)
            "average_loss": each["average_loss"],
        }
    except Exception as error:
        ic(each)
        raise error

chart_data = chart_data.map(agg2)
# convert for the chart
# for every model, add a label, a color, and extract out a list of x/y values
stats = LazyDict()
stats.min = math.inf
stats.max = -math.inf
def agg3(each):
    try:
        stats.min = min(stats.min, *each["correct"])
        stats.max = max(stats.max, *each["correct"])
        return {
            "label": each["model"],
            "backgroundColor": model_color_map[each["model"]],
            "borderColor": model_color_map[each["model"]],
            "color": model_color_map[each["model"]],
            # x and y pairs
            "data": list(zip(each["binary_class"], each["correct"]))
        }
    except Exception as error1:
        ic(each)
        raise error1

chart_data = chart_data.map(agg3)
chart_data.compute() # LiquidData is lazy unless we tell it to evaluate itself
datasets = chart_data.bundles[0]

stats.range = stats.max - stats.min
stats.padding_portion = 7 
stats.padding = stats.range * (stats.padding_portion/100)

# 
# show the data
# 
ss.DisplayCard("chartjs", {
    "type": "line",
    "options": {
        "pointRadius": 3,
        "scales": {
            "y": {
                "min": round(stats.min - stats.padding, ndigits=0),
                "max": round(stats.max + stats.padding, ndigits=0),
            }
        },
        "layout": {
            # "padding": {
            #     "top": 50
            # },
        },
        "plugins": {
            "legend" : {
                "maxWidth": 600,
                # "title": {
                #     "display": True,
                #     "padding": 15,
                #     "text": ""
                # }
            }
        }
    },
    "data": {
        "labels": [0,1,2,3,4,5,6,7,8,9],
        "datasets": datasets,
    }
})