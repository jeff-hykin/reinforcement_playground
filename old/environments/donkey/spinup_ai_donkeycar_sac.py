import os
from sys import platform
from tools.file_system_tools import FS
import subprocess

port = 9091

exe_path = None
if platform == "linux":
    exe_path = FS.local_path("servers/DonkeySimLinux/donkey_sim.x86_64")
elif platform == "darwin":
    exe_path = FS.local_path("servers/DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim")
# download if needed
if not os.path.isfile(exe_path):
    # download the server and overwrite whatever corrupted files existed
    path_to_script = FS.local_path("setup/download_server.sh")
    os.system(path_to_script)
    

subprocess.run([ "python", "-m", "spinup.run", "sac", "--env_name", "donkey-generated-track-v0", "--port", f"{port}", "--exe_path", f"{exe_path}" ])