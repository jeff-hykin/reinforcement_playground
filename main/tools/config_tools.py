import ez_yaml
from tools.file_system_tools import FS
from os.path import isabs, isfile, isdir, join, dirname, basename, exists, splitext, relpath

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
