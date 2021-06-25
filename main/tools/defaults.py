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
import include
import ez_yaml

# import tensorflow as tf
# from stable_baselines.common import set_global_seeds
# from stable_baselines.ppo2.ppo2 import constfn
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# local
from tools.basics import *
from tools.pytorch_tools import *
from tools.dataset_tools import *
from tools.file_system_tools import FS


# 
# config
# 
config = ez_yaml.to_object(file_path=join(dirname(__file__),"config.yaml"))

# 
# paths
# 
PATHS = config["paths"]
# make paths absolute if they're relative
for each_key in PATHS.keys():
    *folders, name, ext = FS.path_pieces(PATHS[each_key])
    # if there are no folders then it must be a relative path (otherwise it would start with the roo "/" folder)
    if len(folders) == 0:
        folders.append(".")
    # if not absolute, then make it absolute
    if folders[0] != "/":
        if folders[0] == '.' or folders[0] == './':
            _, *folders = folders
        PATHS[each_key] = FS.absolute_path(PATHS[each_key])
