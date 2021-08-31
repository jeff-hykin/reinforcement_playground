from tools.file_system_tools import FS


exec(FS.read("./main/tools/record_keeper.py"))
collection = ExperimentCollection("/home/jeffhykin/repos/reinforcement_playground/main/agents/informed_vae/vae_comparison")

# collection = ExperimentCollection("/home/jeffhykin/repos/reinforcement_playground/main/agents/informed_vae/vae_comparison")
# collection.records[0] example:
# {
#     "testing": true,
#     "average_loss": 9.011544585227966e-05,
#     "accuracy": 0.9704,
#     "correct": 9704,
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
#     "binary_class": 9,
#     "transfer_learning_iteration": 0,
#     "test": "binary_mnist",
#     "seed": 1630011706.0022345,
#     "binary_class_order": [
#         0,
#         1,
#         2,
#         3,
#         4,
#         5,
#         6,
#         7,
#         8,
#         9
#     ],
#     "train_test_ratio": [
#         5,
#         1
#     ],
#     "experiment_number": 1,
#     "error_number": 0,
#     "had_error": false,c
#     "experiment_start_time": 1630011706.0022597,
#     "experiment_end_time": 1630012049.1319325,
#     "experiment_duration": 343.1296727657318
# }

exec(FS.read("./main/tools/liquid_data.py"))
def satisfies(each):
    return (
            # no runs with errors allowed
            not each["had_error"]
            # no training (or other) values
            and each["testing"]
            # only for binary_mnist
            and each["test"] == "binary_mnist"
            # make sure binary_class_order didnt change
            # and each["binary_class"] != None
            #and each["binary_class_order"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            # make sure decoder importance didnt change
            # and not (each["model"] == "split" and each["decoder_importance"] != 0.1)
            # # make sure latent_shape is the same
            and (
                   list(each["latent_shape"]or[]) == [30]
                # (for simple_classifier its called mid_shape)
                or list(each["mid_shape"   ]or[]) == [30]
            )
        )

def fix_fresh_class(each):
    return {
        **each,
        "model": each["model"] if not each["fresh"] else "FreshClassifier",
    }

r = [ each for each in collection.records if each["testing"] ] 
data = LiquidData(collection.records).only_keep_if(
        satisfies
    ).map(
        fix_fresh_class
    )

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

from statistics import mean as average
def agg1(records_within_class):
    print('records_within_class = ', records_within_class)
    print('')
    print('records_within_class["model"][0] = ', records_within_class["model"][0])
    print('records_within_class["binary_class"] = ', records_within_class["binary_class"])
    print('records_within_class["binary_class"][0] = ', records_within_class["binary_class"][0])
    print('records_within_class["transfer_learning_iteration"][0] = ', records_within_class["transfer_learning_iteration"][0])
    print('average(records_within_class["correct"]) = ', average(records_within_class["correct"]))
    print('average(records_within_class["accuracy"]) = ', average(records_within_class["accuracy"]))
    print('average(records_within_class["average_loss"]) = ', average(records_within_class["average_loss"]))
    return {
        
        # for each of these, their list elements are all the same so just grab the first
        "model": print(records_within_class) or records_within_class["model"][0],
        "binary_class": records_within_class["binary_class"][0],
        "transfer_learning_iteration": records_within_class["transfer_learning_iteration"][0],
        
        # for these however, we want to average across experiments
        "correct": average(records_within_class["correct"]),
        "accuracy": average(records_within_class["accuracy"]),
        "average_loss": average(records_within_class["average_loss"]),
    }

averaged = bin_class_agg.map(agg1)
averaged.bundles[0][0]
    

model_color_map = {
    "SplitClassifier": 'rgb(0, 92, 192, 0.9)',
    "SimpleClassifier": "rgb(75, 192, 192, 0.9)",
    "FreshClassifier": "rgb(0, 292, 192, 0.9)",
}

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
        print('each = ', each)
        raise error

def agg3(each):
    try:
        return {
            "label": each["model"],
            "backgroundColor": model_color_map[each["model"]],
            "color": model_color_map[each["model"]],
            # x and y pairs
            "data": list(zip(each["binary_class"], each["correct"]))
        }
    except Exception as error:
        print('each = ', each)
        raise error

             # go back to only grouping by model
chart_data = averaged.aggregate(
    # remove the redundant data (e.g. model name)
    ).map(
        agg2
    # convert for the chart
    # for every model, add a label, a color, and extract out a list of x/y values
    ).map(
        agg3
    )

datasets = chart_data.bundles[0]
print('len(datasets) = ', len(datasets))

# 
# show the data
# 
ss.DisplayCard("chartjs", {
    "type": "line",
    "options": {
        "pointRadius": 3,
        "scales": {
            "y": {
                "min": 9700,
                "max": 10000
            }
        }
    },
    "data": {
        "labels": [0,1,2,3,4,5,6,7,8,9],
        "datasets": datasets,
    }
})