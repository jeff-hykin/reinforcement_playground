import os
import gym
import gym_donkeycar
import numpy as np

from tools.file_system_tools import FS

# TODO: make donkeycar purgeable

# 
# setup the server
# 
class Environment(gym.Env):
    """
    Environment(
        map="donkey-generated-track-v0",
    )
    """
    
    def __init__(
        self,
        map="donkey-generated-track-v0",
        port=9091,
        **kwargs,
    ):
        """
        Arguments:
            map: 
                default is "donkey-generated-track-v0"
            port:
                default is 9091
        """
        from sys import platform
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
        
        self._env = gym.make("donkey-generated-track-v0", conf={ "port": port, "exe_path": exe_path, **kwargs })
        
    def reset(self): # DONE
        return self._env.reset()

    def step(self, action): # DONE
        return self._env.step(action)

    def _get_image(self):
        return self.ale.getScreenRGB2()

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        if self._obs_type == "ram":
            return self._get_ram()
        elif self._obs_type == "image":
            img = self._get_image()
        return img

    def render(self, mode="human"):
        img = self._get_image()
        if mode == "rgb_array":
            return img
        elif mode == "human":
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        env.close()

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            "UP": ord("w"),
            "DOWN": ord("s"),
            "LEFT": ord("a"),
            "RIGHT": ord("d"),
            "FIRE": ord(" "),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def clone_state(self):
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """Restore emulator state w/o system state."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """Clone emulator state w/ system state including pseudorandomness.
        Restoring this state will give an identical environment."""
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """Restore emulator state w/ system state including pseudorandomness."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
