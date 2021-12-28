from torch import nn
import gym
import numpy as np
import silver_spectacle as ss
import torch
from super_map import LazyDict
import math
from collections import defaultdict
import functools
from gym.wrappers import AtariPreprocessing
from agent_builders.a2c.baselines_optimizer import RMSpropTFLike
from stable_baselines3.common.vec_env import VecFrameStack

from time import time
import tools.stat_tools as stat_tools
from tools.basics import product, flatten
from tools.debug import debug
from tools.pytorch_tools import Network, layer_output_shapes, opencv_image_to_torch_image, to_tensor, init, forward, Sequential


def fitness_measurement_trend_up(episode_rewards, spike_suppression_magnitude=8, granuality_branching_factor=3, min_bucket_size=5, max_bucket_proportion=0.65):
    # measure: should trend up, more improvement is better, but trend is most important
    # trend is measured at recusively granular levels: default splits of (1/3th's, 1/9th's, 1/27th's ...)
    # the default max proportion (0.5) prevents bucket from being more than 50% of the full list (set to max to 1 to allow entire list as first "bucket")
    recursive_splits_list = stat_tools.recursive_splits(
        episode_rewards,
        branching_factor=granuality_branching_factor,
        min_size=min_bucket_size,
        max_proportion=max_bucket_proportion, 
    )
    improvements_at_each_bucket_level = []
    for buckets in recursive_splits_list:
        bucket_averages = [ stat_tools.average(each_bucket) for each_bucket in buckets if len(each_bucket) > 0 ]
        improvement_at_this_bucket_level = 0
        for prev_average, next_average in stat_tools.pairwise(bucket_averages):
            absolute_improvement = next_average - prev_average
            # pow is being used as an Nth-root
            # and Nth-root is used because we don't care about big spikes
            # we want to measure general improvement, while still keeping the property that more=>better
            if absolute_improvement > 0:
                improvement = math.pow(absolute_improvement, 1/spike_suppression_magnitude)
            else:
                # just mirror the negative values
                improvement = -math.pow(-absolute_improvement, 1/spike_suppression_magnitude)
            
            improvement_at_this_bucket_level += improvement
        average_improvement = improvement_at_this_bucket_level/(len(bucket_averages)-1) # minus one because its pairwise
        improvements_at_each_bucket_level.append(average_improvement)
    # all split levels given equal weight
    return stat_tools.average(improvements_at_each_bucket_level)

good_data = [ 1.0, 0.0, 0.0, 0.0, 1.0, 4.0, 2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0, 0.0, 4.0, 1.0, 1.0, 1.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 2.0, 0.0, 0.0, 0.0, 4.0, 1.0, 2.0, 4.0, 1.0, 1.0, 0.0, 1.0, 9.0, 1.0, 3.0, 1.0, 5.0, 3.0, 2.0, 2.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 5.0, 0.0, 1.0, 0.0, 1.0, 4.0, 3.0, 2.0, 1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 3.0, 0.0, 1.0, 0.0, 3.0, 2.0, 2.0, 0.0, 1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 3.0, 0.0, 2.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 7.0, 1.0, 4.0, 1.0, 0.0, 4.0, 4.0, 6.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 2.0, 0.0, 1.0, 1.0, 6.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 3.0, 2.0, 3.0, 2.0, 6.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 2.0, 2.0, 1.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 4.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 2.0, 0.0, 5.0, 0.0, 0.0, 10.0, 1.0, 0.0, 4.0, 1.0, 4.0, 2.0, 1.0, 0.0, 4.0, 2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 9.0, 0.0, 2.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 9.0, 2.0, 0.0, 9.0, 5.0, 2.0, 4.0, 2.0, 0.0, 0.0, 2.0, 4.0, 0.0, 1.0, 0.0, 5.0, 2.0, 0.0, 4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 9.0, 5.0, 0.0, 11.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 11.0, 0.0, 9.0, 0.0, 2.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 7.0, 2.0, 2.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 0.0, 1.0, 4.0, 6.0, 2.0, 4.0, 2.0, 5.0, 5.0, 1.0, 0.0, 2.0, 5.0, 10.0, 0.0, 1.0, 5.0, 5.0, 0.0, 2.0, 4.0, 4.0, 4.0, 9.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 1.0, 3.0, 10.0, 3.0, 0.0, 0.0, 10.0, 2.0, 0.0, 0.0, 0.0, 5.0, 2.0, 1.0, 3.0, 1.0, 0.0, 4.0, 0.0, 9.0, 2.0, 4.0, 2.0, 2.0, 2.0, 9.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 4.0, 0.0, 9.0, 4.0, 3.0, 0.0, 4.0, 1.0, 0.0, 2.0, 0.0, 2.0, 3.0, 5.0, 5.0, 0.0, 5.0, 2.0, 4.0, 3.0, 1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 8.0, 0.0, 1.0, 0.0, 9.0, 5.0, 2.0, 0.0, 0.0, 2.0, 0.0, 3.0, 13.0, 0.0, 0.0, 5.0, 3.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 6.0, 2.0, 0.0, 5.0, 2.0, 0.0, 3.0, 9.0, 3.0, 2.0, 0.0, 0.0, 0.0, 1.0, 5.0, 4.0, 2.0, 0.0, 0.0, 6.0, 5.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 6.0, 1.0, 0.0, 5.0, 2.0, 4.0, 2.0, 3.0, 0.0, 2.0, 0.0, 5.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 2.0, 6.0, 0.0, 0.0, 4.0, 2.0, 7.0, 1.0, 1.0, 5.0, 5.0, 5.0, 6.0, 9.0, 0.0, 3.0, 5.0, 2.0, 2.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 7.0, 3.0, 0.0, 2.0, 0.0, 1.0, 4.0, 2.0, 5.0, 2.0, 3.0, 14.0, 13.0, 5.0, 0.0, 0.0, 2.0, 6.0, 0.0, 0.0, 9.0, 3.0, 0.0, 0.0, 3.0, 3.0, 3.0, 2.0, 0.0, 2.0, 0.0, 0.0, 1.0, 6.0, 2.0, 1.0, 3.0, 1.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 2.0, 9.0, 2.0, 9.0, 0.0, 2.0, 2.0, 2.0, 2.0, 4.0, 3.0, 0.0, 9.0, 4.0, 1.0, 4.0, 2.0, 5.0, 3.0, 2.0, 1.0, 10.0, 1.0, 2.0, 4.0, 10.0, 0.0, 3.0, 1.0, 1.0, 3.0, 3.0, 3.0, 2.0, 1.0, 0.0, 1.0, 5.0, 5.0, 1.0, 4.0, 1.0, 0.0, 3.0, 3.0, 0.0, 4.0, 2.0, 5.0, 5.0, 1.0, 6.0, 5.0, 5.0, 1.0, 0.0, 4.0, 1.0, 0.0, 1.0, 5.0, 1.0, 3.0, 0.0, 0.0, 2.0, 2.0, 3.0, 3.0, 0.0, 5.0, 9.0, 3.0, 0.0, 5.0, 2.0, 0.0, 2.0, 4.0, 1.0, 2.0, 5.0, 0.0, 5.0, 2.0, 5.0, 0.0, 2.0, 2.0, 3.0, 2.0, 5.0, 16.0, 2.0, 0.0, 2.0, 0.0, 4.0, 3.0, 1.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 9.0, 3.0, 1.0, 6.0, 2.0, 5.0, 6.0, 2.0, 0.0, 2.0, 1.0, 9.0, 2.0, 6.0, 5.0, 1.0, 1.0, 6.0, 0.0, 4.0, 1.0, 3.0, 2.0, 3.0, 1.0, 3.0, 7.0, 1.0, 3.0, 3.0, 7.0, 1.0, 3.0, 3.0, 1.0, 2.0, 2.0, 8.0, 10.0, 0.0, 0.0, 6.0, 4.0, 2.0, 2.0, 9.0, 2.0, 5.0, 6.0, 3.0, 4.0, 3.0, 4.0, 2.0, 1.0, 2.0, 9.0, 1.0, 3.0, 5.0, 1.0, 1.0, 1.0, 0.0, 3.0, 5.0, 4.0, 9.0, 2.0, 4.0, 2.0, 3.0, 1.0, 4.0, 4.0, 7.0, 4.0, 1.0, 2.0, 2.0, 4.0, 4.0, 2.0, 1.0, 3.0, 5.0, 5.0, 3.0, 5.0, 3.0, 6.0, 1.0, 6.0, 16.0, 4.0, 3.0, 6.0, 4.0, 1.0, 7.0, 6.0, 9.0, 5.0, 6.0, 5.0, 2.0, 1.0, 4.0, 3.0, 1.0, 3.0, 3.0, 0.0, 3.0, 4.0, 3.0, 2.0, 1.0, 3.0, 8.0, 10.0, 6.0, 2.0, 1.0, 4.0, 4.0, 2.0, 5.0, 9.0, 5.0, 8.0, 5.0, 4.0, 4.0, 5.0, 8.0, 6.0, 6.0, 1.0, 3.0, 5.0, 12.0, 8.0, 6.0, 4.0, 3.0, 8.0, 5.0, 5.0, 5.0, 11.0, 10.0, 10.0, 11.0, 9.0, 9.0, 5.0, 4.0, 10.0, 6.0, 9.0, 11.0, 14.0, 5.0, 6.0, 4.0, 7.0, 15.0, 4.0, 13.0, 4.0, 5.0, 9.0, 5.0, 9.0, 7.0, 5.0, 12.0, 7.0, 4.0, 8.0, 7.0, 6.0, 5.0, 6.0, 8.0, 2.0, 3.0, 9.0, 6.0, 9.0, 5.0, 5.0, 6.0, 6.0, 13.0, 7.0, 13.0, 3.0, 5.0, 4.0, 12.0, 4.0, 6.0, 13.0, 14.0, 8.0, 9.0, 9.0, 7.0, 5.0, 13.0, 13.0, 7.0, 11.0, 12.0, 8.0, 8.0, 6.0, 6.0, 7.0, 12.0, 8.0, 4.0, 9.0, 6.0, 6.0, 15.0, 9.0, 7.0, 7.0, 11.0, 8.0, 7.0, 21.0, 8.0, 7.0, 8.0, 10.0, 14.0, 7.0, 5.0, 9.0, 7.0, 10.0, 9.0, 9.0, 10.0, 14.0, 14.0, 13.0, 16.0, 12.0, 13.0, 7.0, 14.0, 8.0, 9.0, 17.0, 11.0, 16.0, 10.0, 12.0, 18.0, 9.0, 10.0, 13.0, 18.0, 17.0, 8.0, 8.0, 11.0, 7.0, 14.0, 17.0, 17.0, 12.0, 19.0, 20.0, 12.0, 20.0, 33.0, 18.0, 10.0, 11.0, 9.0, 15.0, 11.0, 17.0, 24.0, 10.0, 24.0, 12.0, 17.0, 16.0, 12.0, 10.0, 9.0, 18.0, 15.0, 19.0, 25.0, 23.0, 17.0, 20.0, 18.0, 31.0, 18.0, 17.0, 23.0, 25.0, 11.0, 20.0, 19.0, 24.0, 17.0, 18.0, 32.0, 23.0, 24.0, 40.0, 30.0, 30.0, 50.0, 24.0, 39.0, 33.0, 11.0, 28.0, 22.0, 24.0, 36.0, 49.0, 50.0, 22.0, 11.0, 19.0, 19.0, 33.0, 44.0, 15.0, 42.0, 22.0, 22.0, 57.0, 49.0, 42.0, 39.0, 58.0, 35.0, 24.0, 34.0, 22.0, 54.0, 32.0, 28.0, 38.0, 27.0, 17.0, 29.0, 55.0, 24.0, 43.0, 39.0, 70.0, 22.0, 30.0, 39.0, 36.0, 37.0, 51.0, 47.0, 59.0, 56.0, 20.0, 40.0, 37.0, 58.0, 47.0, 41.0, 24.0, 72.0, 95.0, 79.0, 59.0, 10.0, 105.0, 35.0, 29.0, 69.0, 88.0, 81.0, 51.0, 78.0, 25.0, 55.0, 48.0, 67.0, 64.0, 47.0, 72.0, 87.0, 26.0, 64.0, 70.0, 125.0, 38.0, 59.0, 44.0, 62.0, 195.0, 120.0, 259.0, 68.0, 81.0, 26.0, 42.0, 71.0, 73.0, 328.0, 196.0, 52.0, 188.0, 47.0, 116.0, 167.0, 79.0, 218.0, 43.0, 68.0, 136.0, 86.0, 96.0, 192.0, 79.0, 64.0, 97.0, 322.0, 140.0, 69.0, 215.0, 118.0, 39.0, 148.0, 85.0, 151.0, 136.0, 109.0, 95.0, 157.0, 192.0, 229.0, 380.0, 260.0, 80.0, 165.0, 132.0, 96.0, 305.0, 181.0, 139.0, 12.0, 150.0, 358.0, 265.0, 91.0, 290.0, 93.0, 145.0, 235.0, 263.0, 222.0, 339.0, 233.0, 269.0, 119.0, 149.0, 301.0, 294.0, 183.0, 302.0, 354.0, 91.0, 172.0, 225.0, 366.0, 97.0, 158.0, 192.0, 149.0, 85.0, 409.0, 347.0, 174.0, 420.0, 81.0, 114.0, 308.0, 238.0, 391.0, 341.0, 213.0, 111.0, 96.0, 362.0, 269.0, 362.0, 377.0, 303.0, 61.0, 109.0, 203.0, 55.0, 349.0, 294.0, 277.0, 95.0, 76.0, 326.0, 318.0, 309.0, 381.0, 299.0, 395.0, 196.0, 339.0, 344.0, 385.0, 333.0, 400.0, 385.0, 348.0, 386.0, 94.0, 286.0, 387.0, 257.0, 394.0, 303.0, 297.0, 65.0, 124.0, 373.0, 405.0, 270.0, 370.0, 48.0, 41.0, 343.0, 388.0, 353.0, 275.0, 348.0, 306.0, 413.0, 95.0, 362.0, 395.0, 399.0, 398.0, 373.0, 315.0, 271.0, 288.0, 204.0, 291.0, 418.0, 165.0, 304.0, 421.0, 70.0, 400.0, 371.0, 428.0, 428.0, 32.0, 226.0, 385.0, 410.0, ]

print(fitness_measurement_trend_up(good_data, min_bucket_size=200))
