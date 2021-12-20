from super_map import LazyDict
from statistics import mean as average
import math

def create_linear_interpolater(from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    return lambda value : (((value - from_min)/from_range) * to_range) + to_min

def filter_outliers(list_of_numbers, cutoff=25, multiplier=1.5, already_sorted=False):
    list_of_numbers = sorted(tuple(list_of_numbers)) if not already_sorted else list_of_numbers
    # needs at least 5 values to compute an IQR
    if len(list_of_numbers) < 4:
        return list_of_numbers
    
    q1 = get_quantile(list_of_numbers, cutoff)
    q3 = get_quantile(list_of_numbers, 100-cutoff)
    
    iqr = q3 - q1
    minimum = q1 - iqr * 1.5
    maximum = q3 + iqr * 1.5

    return [ each for each in list_of_numbers if each >= minimum and each <= maximum ]

def get_quantile(array, quantile):
    import math
    index = quantile / 100.0 * (len(array) - 1)
    # if integer
    if int(index) == index:
        return array[index]
    else:
        lower_index = math.floor(index)
        higher_index = math.ceil(index)
        # interpolate
        linear_interpolater = create_linear_interpolater(lower_index, higher_index, array[lower_index], array[higher_index])
        return linear_interpolater(index)

def get_stats(list_of_numbers: [float]) -> LazyDict:
    """
    Example:
        stats = get_stats([ 1, 4, 24.4, 5, 99 ])
        print(stats.min)
        print(stats.max)
        print(stats.range)
        print(stats.average)
        print(stats.median)
        print(stats.sum)
        print(stats.was_dirty)
        print(stats.cleaned)
        print(stats.count) # len(stats.cleaned)
    """
    import math
    numbers_as_tuple = tuple(list_of_numbers)
    original_length = len(numbers_as_tuple)
    # sort and filter bad values
    list_of_numbers = tuple(
        each
            for each in sorted(numbers_as_tuple)
            if not (each is None or each is math.nan)
    )
    # if empty return mostly bad values
    if len(list_of_numbers) == 0:
        return LazyDict(
            min=None,
            max=None,
            range=0,
            average=0,
            median=None,
            sum=0,
            was_dirty=(original_length != len(list_of_numbers)),
            cleaned=list_of_numbers,
            count=len(list_of_numbers),
        )
    # 
    # Calculate stats
    # 
    # TODO: make this more efficient by making a new class, and having getters 
    #       for each of these values, so that theyre calculated as-needed (and then cached)
    median = list_of_numbers[math.floor(len(list_of_numbers)/2.0)]
    minimum = list_of_numbers[0]
    maximum = list_of_numbers[-1]
    sum_total = 0
    for each in list_of_numbers:
        sum_total += each
    return LazyDict(
        min=minimum,
        max=maximum,
        range=minimum-maximum,
        average=sum_total/len(list_of_numbers),
        median=median,
        sum=sum_total,
        was_dirty=(original_length != len(list_of_numbers)),
        cleaned=list_of_numbers,
        count=len(list_of_numbers),
    )

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
        results.append(sum(average_items)/len(average_items))
    return results

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

def recursive_splits(a_list, branching_factor=2, min_size=2, max_proportion=0.5):
    number_of_splits = int(math.log(len(a_list))/math.log(branching_factor))
    splits = []
    for each_exponential_size in reversed(range(0, number_of_splits)):
        bundle_size = branching_factor**each_exponential_size
        print('bundle_size = ', bundle_size)
        print('bundle_size < min_size = ', bundle_size < min_size)
        print('bundle_size/len(a_list) > max_proportion = ', bundle_size/len(a_list) > max_proportion)
        if bundle_size < min_size:
            break
        if bundle_size/len(a_list) > max_proportion:
            continue
        splits.append(tuple(bundle(a_list, bundle_size=bundle_size)))
    return splits

# def savitzky_golay_smoothing(a_list, strength):
#     from SGCC.savgol import get_coefficients # pip install savgol-calculator
#     import math
#     size = len(a_list)
#     if size < 3:
#         return a_list
#     # map strength logarithmically to percentages
#     strength_scaled = math.ceil(math.log(create_linear_interpolater(0, 1, 0, 2**size)(strength))/2)*2
# 
#     derivative = 0
#     cubic_polynomial = 3
#     coefficients = get_coefficients(
#         smoothing=derivative,
#         order=cubic_polynomial,
#         window_size=strength_scaled,
#         offset=0
#     )
#     normalizer = sum(coefficients)
#     half_window_size = window_size/2
#     rolling_average = []
#     for index, each in enumerate(a_list):
#         backwards_index = index*window_size - half_window_size
#         forwards_index  = index*window_size + half_window_size
#         in_the_middle = backwards_index <= 0 and forwards_index < len(a_list)
#         if in_the_middle:
#             nearby_numbers = a_list[backwards_index:forwards_index]
#             rolling_average.append(
#                 sum([
#                     coefficient * number
#                         for coefficient, number in zip(coeffs, nearby_numbers)
#                 ]) / normalizer
#             )
#         elif 
    