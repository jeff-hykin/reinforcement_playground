import pandas as pd
import tools.stat_tools as stat_tools

data = {
    "fitnesses": [ 
        1.22,
        0.97,
        1.04,
        1.34,
        1.29,
        1.2,
        1.2,
        1.26,
        1.46,
        1.14,
        1.04,
        1.29,
        1.41,
        1.21,
        1.25,
        1.26,
        1.29,
        0.96,
        1.08,
        1.19,
        1.08,
        1.15,
        1.43,
        1.16,
        1.33,
        1.08,
        1.31,
        1.35,
        1.26,
        1.46,
        1.29,
        1.08,
        1.38,
    ],
    "discount_factors" :[
        0.98,
        0.91,
        0.89,
        0.99,
        0.83,
        0.96,
        0.87,
        0.81,
        0.92,
        0.87,
        0.93,
        1.00,
        0.94,
        0.94,
        0.91,
        0.96,
        0.85,
        0.91,
        0.94,
        0.89,
        0.97,
        1.00,
        0.93,
        0.93,
        0.93,
        0.95,
        0.90,
        0.92,
        0.96,
        0.97,
        0.98,
        0.97,
        0.94,
    ],
    "actor_learning_rate": [
        5.42927230639307E-05,
        1.00058605261015E-05,
        0.000676602647009185,
        0.000121534481300853,
        0.0039777228535635,
        0.038284278150746,
        0.000252350824679718,
        1.33327552198693E-05,
        0.000823479929811354,
        0.0629015305387879,
        0.00503263128343933,
        0.000358757430869932,
        8.57360205561148E-05,
        0.00246800612710367,
        5.09678186069722E-05,
        0.0159919887914878,
        0.00137325609619237,
        4.74417237083403E-05,
        0.000194617703174905,
        0.000863772656122294,
        0.000112399025105017,
        0.000119295967481013,
        2.42403605946825E-05,
        2.98366986139268E-05,
        2.51682102073444E-05,
        0.000336449545288091,
        2.074648831549E-05,
        6.65363086725072E-05,
        0.0124789610178439,
        0.000640622634616152,
        0.00166871605536648,
        0.000555404832454045,
        7.24116186125872E-05,
    ],
    "critic_learning_rate": [
        0.000407119631883438,
        0.00203225666023996,
        0.0388153224156872,
        6.00789313856603E-05,
        2.74752439133975E-05,
        0.000123413258775151,
        0.0372808694454963,
        0.000447426384819328,
        0.0071511749586473,
        2.21980169164294E-05,
        0.00507843352130538,
        0.00743530071862647,
        7.74197643077547E-05,
        0.012595318572055,
        0.000913062186622161,
        0.000190799434242641,
        1.07166452847228E-05,
        0.0927828044554328,
        0.00162941525389212,
        0.00396094732358576,
        0.013583068656025,
        7.55326043643941E-05,
        6.58539664645445E-05,
        0.00026740933632396,
        3.82672660074669E-05,
        1.21894424736838E-05,
        0.000864812791072578,
        9.80518515834978E-05,
        4.57795254054915E-05,
        0.000340993093175542,
        0.000453265658602873,
        0.000171955621570943,
        0.000438496372637853,
    ],
}
df = pd.DataFrame(data)
# 
# rolled average discounts
# 
def associated_rolled_average(df, metric_key, sort_key, window=10):
    discount_sorted_df = df.sort_values(sort_key)
    discount_fitnesses, discount_factors = (discount_sorted_df[metric_key].tolist(), discount_sorted_df[sort_key].tolist())
    discount_rolled_fitnesses = stat_tools.rolling_average(discount_fitnesses, window=window)
    discount_rollled_factors = stat_tools.rolling_average(discount_factors, window=window)
    for each1,each2 in zip(discount_rolled_fitnesses, discount_rollled_factors): print(f"{each1}, {each2}")

def associated_buckets_average(df, metric_key, sort_key, window=5, shift=0):
    sorted_df = df.sort_values(sort_key)
    metric_values, sort_values = (sorted_df[metric_key].tolist(), sorted_df[sort_key].tolist())
    
    first_metric_bucket = metric_values[0:shift]
    metric_values = metric_values[shift:]
    
    first_sort_bucket = sort_values[0:shift]
    sort_values = sort_values[shift:]
    
    if shift > 0:
        metric_averages = [ stat_tools.average(first_metric_bucket) ] + [ stat_tools.average(each) for each in stat_tools.bundle(metric_values, bundle_size=window)]
        sort_averages   = [ stat_tools.average(first_sort_bucket)   ] + [ stat_tools.average(each) for each in stat_tools.bundle(sort_values, bundle_size=window)]
    else:
        metric_averages = [ stat_tools.average(each) for each in stat_tools.bundle(metric_values, bundle_size=window)]
        sort_averages   = [ stat_tools.average(each) for each in stat_tools.bundle(sort_values, bundle_size=window)]
        
    for each1, each2 in zip(metric_averages, sort_averages): print(f"{each1}, {each2}")
    return metric_averages, sort_averages

def 

associated_rolled_average(df, "fitnesses", "discount_factors", window=10)
associated_rolled_average(df, "fitnesses", "actor_learning_rate", window=5)
associated_buckets_average(df, "fitnesses", "actor_learning_rate", window=5, shift=1)
associated_buckets_average(df, "fitnesses", "actor_learning_rate", window=5, shift=2)