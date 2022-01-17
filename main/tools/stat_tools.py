import math
import scipy.stats as stats
from statistics import mean as average
from statistics import stdev as standard_deviation
from super_map import LazyDict

def probabilitity_of_at_least_one(*probabilities):
    chance_of_none = 1
    for each in probabilities:
        chance_of_none *= 1 - each
    return 1 - chance_of_none

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

def pairwise(an_iterable):
    # grab the first one
    iterable = (each for each in an_iterable)
    prev = next(iterable)
    for current in iterable:
        yield prev, current
        prev = current

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

def recursive_splits(a_list, branching_factor=2, min_size=2, max_proportion=0.65):
    list_length = len(a_list)
    number_of_splits = int(math.log(list_length)/math.log(branching_factor))
    splits = []
    for each_exponential_size in reversed(range(0, number_of_splits)):
        bundle_size = math.ceil(list_length/(branching_factor**each_exponential_size))
        if bundle_size < min_size:
            continue
        if bundle_size/list_length > max_proportion:
            continue
        splits.append(tuple(bundle(a_list, bundle_size=bundle_size)))
    return tuple(reversed(splits))

def increasingly_strict_confidence(
        sample_size,
        shortest_interval=0.3, # this means when the sample is big, there will be a 30% confidence interval requirement
        strictness_rate=1,  # bigger=get to the shortest_interval faster. at strictness_rate=1, sample_size=30, confidence will be around 95%
    ):
    # starts at maximum = 1
    # as x gets bigger 
    # approches shortest_interval
    slowness = 0.006 * strictness_rate # the 0.006 was solved-for to make confidence ~95% for sample size of 30
    offsetter = 2 # anything smaller than this will have a 100% confindence interval (e.g. any/all results are within the interval)
    if sample_size <= offsetter:
        return 1
    else:
        shifted_to_the_right         = sample_size - offsetter
        streched_out_curve           = shifted_to_the_right * slowness 
        scaled_from_one_half_to_zero = 2/( 1 + math.exp(streched_out_curve) )
        # range is 1^
        new_range = 1 - shortest_interval
        rescaled_curve = scaled_from_one_half_to_zero * new_range
        shifted_up     = rescaled_curve + shortest_interval
        return shifted_up

def confirmed_outstandingly_low(item, existing_items, shortest_interval=0.3, strictness_rate=1):
    purified_existing_items = tuple(to_pure(each) for each in existing_items)
    
    confidence_needed = increasingly_strict_confidence(
        sample_size=len(purified_existing_items),
        shortest_interval=shortest_interval,
        strictness_rate=strictness_rate,
    )
    the_mean = average(purified_existing_items)
    standard_deviation_amount = standard_deviation(purified_existing_items)
    number_of_standard_deviations_needed = stats.norm.ppf(confidence_needed)
    cutoff_point = the_mean - (number_of_standard_deviations_needed * standard_deviation_amount)
    if to_pure(item) < cutoff_point:
        return True
    else:
        return False

def probability_of_belonging_if_bellcurve(item, existing_items, above=False, below=False):
    the_mean = average(existing_items)
    standard_deviation_amount = standard_deviation(existing_items)
    how_many_deviations_away = math.abs(item-the_mean) / standard_deviation_amount
    return stats.norm.cdf(how_many_deviations_away)
    

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