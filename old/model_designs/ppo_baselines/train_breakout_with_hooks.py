import gym
import os
import numpy as np
import matplotlib.pyplot as plt
import silver_spectacle as ss

from stable_baselines3 import PPO
from old.model_designs.ppo_baselines.breakout import env


from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

# 
# setup Env
# 
log_dir = "logs/stable_baselines/"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env,log_dir)

# 
# 
# setup Model
# 
# 

model = PPO("MlpPolicy", env, verbose=1)
# 
# setup hooks
# 
class Hooks(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(Hooks, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'hook_data')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True

# 
# Train the agent
# 
timesteps = 100000
model.learn(
    total_timesteps=int(timesteps),
    callback=Hooks(
        check_freq=1000,
        log_dir=log_dir
    ),
)

# 
# display
# 
import pandas
df = pandas.read_csv(log_dir+"/monitor.csv", header=0, skiprows=[0])
rewards = df["r"]
losses = df["l"]
timesteps = [ int(each) for each in df["t"] ] # these are floats... I'm not sure what the units are (its not arbitrary)
ss.DisplayCard("quickLine", list(zip(timesteps, rewards)))
ss.DisplayCard("quickLine", list(zip(timesteps, losses)))