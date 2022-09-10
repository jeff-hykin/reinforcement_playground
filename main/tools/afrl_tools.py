def average(iterable):
    from statistics import mean
    from trivial_torch_tools.generics import to_pure
    return mean(tuple(to_pure(each) for each in iterable))

def median(iterable):
    from statistics import median
    from trivial_torch_tools.generics import to_pure
    return median(tuple(to_pure(each) for each in iterable))

def stats(number_iterator):
    import math
    from super_map import LazyDict
    from statistics import stdev, median, quantiles
    from trivial_torch_tools.generics import to_pure
    
    minimum = math.inf
    maximum = -math.inf
    total = 0
    values = [] # for iterables that get consumed
    for each in number_iterator:
        values.append(to_pure(each))
        total += each
        if each > maximum:
            maximum = each
        if each < minimum:
            minimum = each
    
    count = len(values)
    range = maximum-minimum
    average     = total / count     if count != 0 else None
    median      = median(values)    if count != 0 else None
    stdev       = stdev(values)     if count  > 1 else None
    normalized  = tuple((each-minimum)/range for each in values) if range != 0 else None
    (q1,_,q3),_ = quantiles(values) if count  > 1 else (None,None,None),None
    
    return LazyDict(
        max=maximum,
        min=minimum,
        range=range,
        count=count,
        sum=total,
        average=average,
        stdev=stdev,
        median=median,
        q1=q1,
        q3=q3,
        normalized=normalized,
    )    

def simple_stats(number_iterator):
    import math
    from super_map import LazyDict
    from statistics import stdev, median, quantiles
    from trivial_torch_tools.generics import to_pure
    
    minimum = math.inf
    maximum = -math.inf
    total = 0
    values = [] # for iterables that get consumed
    for each in number_iterator:
        values.append(to_pure(each))
        total += each
        if each > maximum:
            maximum = each
        if each < minimum:
            minimum = each
    
    count = len(values)
    range = maximum-minimum
    average     = total / count     if count != 0 else None
    median      = median(values)    if count != 0 else None
    stdev       = stdev(values)     if count  > 1 else None
    
    return LazyDict(
        max=maximum,
        min=minimum,
        range=range,
        count=count,
        sum=total,
        average=average,
        stdev=stdev,
        median=median,
    )    

def to_numpy(value):
    import torch
    import numpy
    from trivial_torch_tools.generics import to_pure
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    else:
        return numpy.array(to_pure(value))

def flatten(ys):
    return [x for xs in ys for x in xs]

def get_discounted_rewards(rewards, gamma):
    import torch
    from trivial_torch_tools import to_tensor
    rewards   = to_tensor(rewards)
    timesteps = to_tensor(range(len(rewards)))
    gammas    = to_tensor(gamma for each in timesteps)
    return rewards * (gammas ** timestep)

def normalize(values, max, min):
    """
    all elements of the output should be between 0 and 1
    """
    reward_range = max - min
    return tuple((each - min)/reward_range for each in values)

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]

def minibatch(batch_size, *data):
    import numpy as np
    indices = np.arange(len(data[0]))
    np.random.shuffle(indices)
    for batch_ind in divide_chunks(indices, batch_size):
        yield [datum[batch_ind] for datum in data]

def bundle(iterable, bundle_size):
    next_bundle = []
    for each in iterable:
        next_bundle.append(each)
        if len(next_bundle) >= bundle_size:
            yield tuple(next_bundle)
            next_bundle = []
    # return any half-made bundles
    if len(next_bundle) > 0:
        yield tuple(next_bundle)

def rolling_average(a_list, window):
    results = []
    if len(a_list) < window * 2:
        return a_list
    near_the_end = len(a_list) - 1 - window 
    for index, each in enumerate(a_list):
        # at the start
        if index < window:
            average_items = a_list[0:index]+a_list[index:index+window]
        # at the end
        elif index > near_the_end:
            average_items = a_list[index-window:index]+a_list[index:len(a_list)]
        else:
            # this could be done a lot more efficiently with a rolling sum, oh well! ¯\_(ツ)_/¯ 
            average_items = a_list[index-window:index+window+1]
        # fallback
        if len(average_items) == 0:
            average_items = [ a_list[index] ]
        results.append(sum(average_items)/len(average_items))
    return results

def log_scale(number):
    import math
    if number > 0:
        return math.log(number+1)
    else:
        return -math.log((-number)+1)

def train_test_split(*args, split_proportion):
    import numpy as np
    from trivial_torch_tools import to_tensor
    def split(data, indices, train_pct=0.66):
        div = int(len(data) * train_pct)
        train, test = indices[:div], indices[div:]
        return to_tensor(data[train]), to_tensor(data[test])
    
    indices = np.arange(len(args[0]))
    output = []
    for each in args:
        output.append(split(each, indices, split_proportion))
    
    return output

colors = dict(
    yellow=       '#fec355',
    light_yellow= '#ddd790',
    lime=         '#c3e88d',
    green=        '#4ec9b0',
    light_blue=   '#89ddff',
    blue=         '#82aaff',
    deep_blue=    '#00aeff',
    purple=       '#c792ea',
    pink=         '#e57eb3',
    red=          '#f07178',
)
def wrap_around_get(number, a_list):
    list_length = len(a_list)
    return a_list[((number % list_length) + list_length) % list_length]

def get_color(index):
    return wrap_around_get(index, list(colors.values()))

def multi_line_plot(a_dict, path, x_axis_label, y_axis_label):
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    for index, (line_name, line_points) in enumerate(a_dict.items()):
        x_values = tuple(x for x,y in line_points)
        y_values = tuple(y for x,y in line_points)
        color = get_color(index=index)
        plt.plot(x_values, y_values, marker='.', color=color, label=line_name)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()
    plt.savefig(
        FS.clear_a_path_for(
            path,
            overwrite=True
        )
    )
    return plt

def key_prepend(key, a_dict):
    """
    key_prepend("reward", stats(rewards))
    # { "max": , "min": , ... }
    # =>
    # { "reward_max": , "reward_min": , ... }
    """
    new_dict = {}
    for each_key, each_value in a_dict.items():
        new_key = f"{key}_{each_key}"
        new_dict[new_key] = each_value
    return new_dict

from torch import nn
def feed_forward(layer_sizes, activation=nn.Tanh, output_activation=nn.Identity):
    from trivial_torch_tools import Sequential
    layers = []
    for index in range(len(layer_sizes) - 1):
        activation_class = activation if index < len(layer_sizes) - 2 else output_activation
        layers += [
            nn.Linear(layer_sizes[index], layer_sizes[index + 1]),
            activation_class(),
        ]
    return Sequential(*layers)

def log_graph(data):
    import silver_spectacle as ss
    colors = [
        'rgb(75, 192, 192, 0.5)',
        'rgb(0, 292, 192, 0.5)',
        'rgb(0, 92, 192, 0.5)',
        'rgb(190, 92, 192, 0.5)',
        'rgb(75, 192, 192, 0.5)',
        'rgb(0, 292, 192, 0.5)',
        'rgb(0, 92, 192, 0.5)',
        'rgb(190, 92, 192, 0.5)',
        'rgb(75, 192, 192, 0.5)',
        'rgb(0, 292, 192, 0.5)',
        'rgb(0, 92, 192, 0.5)',
        'rgb(190, 92, 192, 0.5)',
        'rgb(75, 192, 192, 0.5)',
        'rgb(0, 292, 192, 0.5)',
        'rgb(0, 92, 192, 0.5)',
        'rgb(190, 92, 192, 0.5)',
    ]
    ss.DisplayCard("chartjs", {
        "type": 'line',
        "data": {
            "datasets": [
                {
                    "label": each_key,
                    "data": ({"x":x, "y":y} for x,y in each_value),
                    "tension": 0.1,
                    "backgroundColor": colors.pop(),
                }
                    for each_key, each_value in data.items()
            ]
        },
        "options": {
            "pointRadius": 3, # the size of the dots
            "scales": {
                "y": {
                    "type": "logarithmic",
                },
            }
        }
    })

class WeightUpdate(object):
    """
    with WeightUpdate(optimizer=self.optimizer) as step:
        step.loss = self.loss_function()
    """
    def __init__(self, *, optimizer):
        self.optimizer = optimizer
        self.loss = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, _, error, traceback):
        if error is not None:
            # error cleanup HERE
            raise error
            
        if self.optimizer:
            self.optimizer.zero_grad()
            if not (self.loss is None):
                self.loss.backward()
            self.optimizer.step()

def confidence_interval(confidence_percent, samples):
    import statistics
    import scipy.stats as st
    min_value, max_value = st.t.interval(alpha=confidence_percent/100, df=len(samples)-1, loc=statistics.mean(samples), scale=st.sem(samples)) 
    return min_value, max_value

def confidence_interval_size(confidence_percent, samples):
    import statistics
    import scipy.stats as st
    min_value, max_value = st.t.interval(alpha=confidence_percent/100, df=len(samples)-1, loc=statistics.mean(samples), scale=st.sem(samples)) 
    return abs(max_value-min_value)/2
    
def probability_of_belonging_if_bellcurve(item, bellcurve_mean, bellcurve_stdev):
    import scipy.stats as stats
    import math
    how_many_deviations_away = abs(item-bellcurve_mean) / bellcurve_stdev
    return stats.norm.cdf(how_many_deviations_away)

def jenson_shannon_divergence(net_1_logits, net_2_logits):
    from torch.functional import F
    net_1_probs =  F.softmax(net_1_logits, dim=0)
    net_2_probs=  F.softmax(net_2_logits, dim=0)
    
    total_m = 0.5 * (net_1_probs + net_1_probs)
    
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduction="batchmean") 
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduction="batchmean") 
    return (0.5 * loss)

def save_all_charts_to(path, overwrite=True):
    import requests
    import file_system_py as FS
    FS.clear_a_path_for(path, overwrite=overwrite)
    FS.write(
        data=requests.get(url='http://localhost:9900/').text,
        to=path,
    )


def multi_plot(data, vertical_label=None, horizonal_label=None, title=None, color_key={}):
    import silver_spectacle as ss
    datasets = []
    labels = {}
    for each_key, each_line in data.items():
        values = []
        for x, y in each_line:
            labels[x] = None
            values.append(y)
        
        datasets.append(dict(
            label=each_key,
            data=values,
            fill=False,
            tension=0.4,
            cubicInterpolationMode='monotone',
            backgroundColor=color_key.get(each_key, 'rgb(0, 292, 192, 0.5)'),
            borderColor=color_key.get(each_key, 'rgb(0, 292, 192, 0.5)'),
            color=color_key.get(each_key, 'rgb(0, 292, 192, 0.5)'),
        ))
        
    
    labels = list(labels.keys())
    return ss.DisplayCard("chartjs", {
        "type": 'line',
        "data": {
            "labels": labels,
            "datasets": datasets,
        },
        "options": {
            "plugins": {
                "title": {
                    "display": (not (not title)),
                    "text": title,
                }
            },
            "pointRadius": 3, # the size of the dots
            "scales": {
                "x": {
                    "title": {
                        "display": horizonal_label,
                        "text": horizonal_label,
                    },
                },
                "y": {
                    "title": {
                        "display": vertical_label,
                        "text": vertical_label,
                    },
                    # "min": 50,
                    # "max": 100,
                },
            }
        }
    })

def multi_variance_plot(data, deviations=None, vertical_label=None, horizonal_label=None, title=None, color_key={}):
    import silver_spectacle as ss
    from trivial_torch_tools.generics import to_pure
    
    from statistics import mean as average
    from statistics import stdev
    
    def is_hex_string(string):
        import re
        match = re.match(r'^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})?$', string.lower())
        return not not match
    
    def hex_string_to_rgb_list(string):
        import re
        match = re.match(r'^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})?$', string.lower())
        if match:
            base_rgb = [ int(match[1], base=16), int(match[2], base=16), int(match[3], base=16) ]
            if match[4]:
                base_rgb.append(int(match[4], base=16))
            return base_rgb
    
    def rgb_string_to_rgb_list(string):
        import re
        match = re.match(r'^rgb\( *([1-2]?\d{1,2}) *, *([1-2]?\d{1,2}) *, *([1-2]?\d{1,2}) *(, *(?:\d+(?:\.\d+)?|\d*\.\d+))?\)$', string)
        if match:
            base_rgb = [ int(match[1]), int(match[2]), int(match[3]) ]
            if match[4]:
                base_rgb.append(float(match[4]))
            return base_rgb
    
    datasets = []
    labels = {}
    for each_key, each_line in data.items():
        color = color_key.get(each_key, 'rgb(0, 292, 192, 0.5)')
        
        # create lighter color
        color_as_rgb_list = hex_string_to_rgb_list(color) if is_hex_string(color) else rgb_string_to_rgb_list(color)
        if len(color_as_rgb_list) == 3:
            color_as_rgb_list.push(1)
        color_as_rgb_list[3] = color_as_rgb_list[3] / 2
        lighter_color = f'''rgb({",".join(color_as_rgb_list)})'''
        
        values = []
        for x, y in each_line:
            labels[x] = None
            values.append(to_pure(y))
        
        averages = values
        if type(deviations) == type(None):
            averages   = tuple(average(each) for each in values)
            deviations = tuple(stdev(each) for each in values)
            
        tension = 0.4
        # centerline
        datasets.append(dict(
            label=each_key,
            data=averages,
            fill=False,
            tension=tension,
            color=color,
            borderColor=color,
            backgroundColor=color,
        ))
        # lowerbound
        datasets.append(dict(
            data=tuple(each_average-each_deviation for each_average, each_deviation in zip(averages, deviations)),
            tension=tension,
            label= '',
            fill= '-1',
            pointBackgroundColor= 'rgba(0, 0, 0, 0.0)',
            pointBorderColor= 'rgba(0, 0, 0, 0.0)',
            pointBorderWidth= 1,
            borderColor= 'rgba(0, 0, 0, 0.0)',
            backgroundColor= lighter_color,
            borderWidth= 1,
        ))
        # UpperBound
        datasets.append(dict(
            data=tuple(each_average+each_deviation for each_average, each_deviation in zip(averages, deviations)),
            tension=tension,
            label= '',
            fill= '-2',
            pointBackgroundColor= 'rgba(0, 0, 0, 0.0)',
            pointBorderColor= 'rgba(0, 0, 0, 0.0)',
            pointBorderWidth= 1,
            borderColor= 'rgba(0, 0, 0, 0.0)',
            backgroundColor= lighter_color,
            borderWidth= 1,
        ))
        
    labels = list(labels.keys())
    return ss.DisplayCard("chartjs", {
        "type": 'line',
        "data": {
            "labels": labels,
            "datasets": datasets,
        },
        "options": {
            "plugins": {
                "title": {
                    "display": (not (not title)),
                    "text": title,
                }
            },
            "pointRadius": 3, # the size of the dots
            "scales": {
                "x": {
                    "title": {
                        "display": horizonal_label,
                        "text": horizonal_label,
                    },
                },
                "y": {
                    "title": {
                        "display": vertical_label,
                        "text": vertical_label,
                    },
                    # "min": 50,
                    # "max": 100,
                },
            }
        }
    })
# 
# override print
# 
real_print = print
def print(*args, to_string=False, **kwargs): # print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)
    from io import StringIO
    if to_string:
        string_stream = StringIO()
        # dump to string
        real_print(*args, **{ "flush": True, **kwargs, "file":string_stream })
        output_str = string_stream.getvalue()
        string_stream.close()
        return output_str
        
    if hasattr(print, "disable") and print.disable:
        return
        
    if hasattr(print, "indent"):
        if print.indent > 0:
            indent = print.indent_string*print.indent
            # dump to string
            output_str = print(*args, **{ **kwargs, "to_string":True})
            # indent it
            output_str = indent+output_str.replace("\n", "\n"+indent)[0:-len(indent)]
            # print it
            return real_print(output_str, **{ "flush": print.flush, **kwargs, "end":""}) 
    
    return real_print(*args, **{ "flush": print.flush, **kwargs})
print.indent_string = "    "
print.indent = 0
print.flush = True
print.disable = False

# @indent_prints
def indent_prints(function_being_wrapped):
    def wrapper(*args, **kwargs):
        original_value = print.indent
        print.indent += 1
        output = function_being_wrapped(*args, **kwargs)
        print.indent = original_value
        return output
    return wrapper

def log_func(function_being_wrapped):
    def wrapper(*args, **kwargs):
        original_value = print.indent
        if hasattr(function_being_wrapped, "__name__"):
            print(function_being_wrapped.__name__)
        print.indent += 1
        output = function_being_wrapped(*args, **kwargs)
        print.indent = original_value
        return output
    return wrapper

# with indent: print("howdy")
class Indent(object):
    """
    with indent:
        print("howdy1")
        with indent:
            print("howdy2")
        print("howdy3")
    """
    def __init__(self, *args, **kwargs):
        self.indent_before = []
    
    def __enter__(self):
        self.indent_before.append(print.indent)
        print.indent += 1
        return print
    
    def __exit__(self, _, error, traceback):
        # restore prev indent
        print.indent = self.indent_before.pop()
        if error is not None:
            # error cleanup HERE
            raise error
indent = Indent()

# with block("staring iterations"):
def block_indent(*args):
    print(*args)
    return indent