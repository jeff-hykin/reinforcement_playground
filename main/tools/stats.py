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