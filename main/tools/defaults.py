# builtin
import argparse
import os
from os.path import isabs, isfile, isdir, join, dirname, basename, exists, splitext, relpath
from os import remove, getcwd, makedirs, listdir, rename, rmdir, system
from collections import OrderedDict
from pprint import pprint
import sys

# external
import numpy as np
import gym
from stable_baselines.common import set_global_seeds
from stable_baselines.ppo2.ppo2 import constfn
import include
import ez_yaml
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# local
from tools.basics import *

config = ez_yaml.to_object(file_path=join(dirname(__file__),"config.yaml"))