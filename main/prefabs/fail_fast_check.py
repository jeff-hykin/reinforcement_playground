from tools.stat_tools import confirmed_outstandingly_low, increasingly_lenient_confidence, probability_of_belonging_if_bellcurve, average, standard_deviation, probabilitity_of_at_least_one


def is_significantly_below_other_curves(current_curve, curves):
    existing_items = tuple(sum(each[0:len(current_curve)]) for each in curves)
    item = sum(current_curve)
    
    confidence_from_rewards_so_far = increasingly_strict_confidence(
        sample_size=len(current_curve),
    )
    confidence_from_existing_items = increasingly_strict_confidence(
        sample_size=len(existing_items)
    )
    
    confidence = probabilitity_of_at_least_one(
        confidence_from_rewards_so_far,
        confidence_from_average_rewards_of_other_runs,
    )
    
    number_of_standard_deviations_needed = stats.norm.ppf(confidence)
    the_mean = average(existing_items)
    standard_deviation_amount = standard_deviation(existing_items)
    cutoff_point = the_mean - (number_of_standard_deviations_needed * standard_deviation_amount)
    
    if item < cutoff_point:
        return True
    else:
        return False