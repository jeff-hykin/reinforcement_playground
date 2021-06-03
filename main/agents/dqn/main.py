import os
import random
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.losses import huber_loss
import numpy as np
from lib import plotting
import random
from collections import defaultdict

class DQN:
    def __init__(self, env, options):
        assert str(env.action_space).startswith("Discrete") or str(
            env.action_space
        ).startswith("Tuple(Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, options)

        self.replay_buffer = []
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.actions = range(self.action_size)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.minibatch_size = 100
        self.index = 0
    
    def on_episode_start(self):
        """
        (optional)
        called once per episode for any init/reset or saving of model checkpoints
        """
        
        self.chance_of_random_action = self.options.epsilon
        self.actions = self.actions
        self.replay_buffer = self.replay_buffer
        self.minibatch_size = self.options.batch_size
        self.frequency_of_refreshing_weights = self.options.update_target_estimator_every
        self.replay_memory_size = self.options.replay_memory_size
        self.discount_of_future_rewards = self.options.gamma

        self.inputs_at = defaultdict(lambda: None)
        self.action_taken_at = defaultdict(lambda: None)
        self.reward_recevied_at = defaultdict(lambda: None)
        
        self.current_time = -1
    
    def decide(observation, reward, is_last_timestep):
        """
        returns an action
        """
        
        #
        # record outcome of previous action
        #
        # initial observation
        if self.current_time == -1:
            self.inputs_at[0] = np.matrix(observation)
        else:
            self.inputs_at[self.current_time+1] = reward
            self.reward_recevied_at[self.current_time] = reward
            self.episode_is_over = is_last_timestep
        
            self.index += 1
            # because tf is dumb
            self.inputs_at[self.current_time + 1] = np.matrix(self.inputs_at[self.current_time + 1])

            #
            # save in buffer
            #
            self.was_not_terminal = 0 if episode_is_over else 1
            self.replay_buffer.append(
                (
                    self.inputs_at[self.current_time],
                    self.action_taken_at[self.current_time],
                    self.reward_recevied_at[self.current_time],
                    self.inputs_at[self.current_time + 1],
                    self.was_not_terminal,
                )
            )
            if len(self.replay_buffer) > self.replay_memory_size:
                # remove the oldest memory once the buffer is full
                self.replay_buffer.pop(0)

            #
            # sample transitions
            #
            # TODO: vectorize this
            batch_inputs = []
            batch_outputs = []
            if self.minibatch_size <= len(self.replay_buffer):
                for (
                    each_input,
                    each_action,
                    each_reward,
                    each_next_input,
                    was_not_terminal,
                ) in random.sample(replay_buffer, k=self.minibatch_size):
                    batch_inputs.append(each_input)
                    batch_outputs.append(
                        each_reward
                        + was_not_terminal
                        * discount_of_future_rewards
                        * self.target_model.predict(
                            x=np.matrix(self.inputs_at[self.current_time])
                        )
                        #  In line 378, only the expected Q-value for the selected action should be obtained from the target network
                        #  The expected Q-values for the rest of the actions should come from the current network and not the target network.
                    )

                #
                # update estimate
                #
                self.model.fit(
                    x=np.squeeze(np.array(batch_inputs)),
                    y=np.squeeze(np.array(batch_outputs)),
                    verbose=0,
                )

            #
            # re-align θʹ = θ
            #
            if self.index % self.frequency_of_refreshing_weights == 0:
                self.update_target_model()
                
        #
        # pick an action
        #
        # chance of taking random action
        if self.chance_of_random_action > random.random():
            self.action_taken_at[self.current_time] = random.sample(actions, k=1)[0]
        # take a calculated action
        else:
            self.action_taken_at[self.current_time] = np.argmax(
                self.model.predict(
                    x=np.matrix(self.inputs_at[self.current_time]), batch_size=1
                )[0]
            )
        
        return self.action_taken_at[self.current_time]
        
    def on_clean_up(self):
        """
        only called once, and should save checkpoints and cleanup any logging info
        """
        return
        
    def _build_model(self):
        layers = self.options.layers
        # Neural Net for Deep-Q learning Model
        model = Sequential()

        model.add(Dense(layers[0], input_dim=self.state_size, activation="relu"))
        if len(layers) > 1:
            for l in layers[1:]:
                model.add(Dense(l, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=huber_loss, optimizer=Adam(lr=self.options.alpha))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def make_epsilon_greedy_policy(self):
        """
        Creates an epsilon-greedy policy based on the given Q-function approximator and epsilon.

        Returns:
            A function that takes a state as input and returns a vector
            of action probabilities.
        """
        nA = self.env.action_space.n

        def policy_fn(state):
            ε, 𝛾, 𝝰, A = (
                self.options.epsilon,
                self.options.gamma,
                self.options.alpha,
                list(range(self.env.action_space.n)),
            )
            greedy_action = self.model.predict(state)
            uniform_probability_value = ε / len(A)
            return [
                (1 - ε) + uniform_probability_value
                if a == greedy_action
                else uniform_probability_value
                for a in A
            ]

        return policy_fn

    def __str__(self):
        return "DQN"

    def plot(self, stats):
        plotting.plot_episode_stats(stats)

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """
        nA = self.env.action_space.n

        def policy_fn(state):
            ε, 𝛾, 𝝰, A = (
                self.options.epsilon,
                self.options.gamma,
                self.options.alpha,
                list(range(self.env.action_space.n)),
            )
            actions = self.model.predict(x=np.matrix(state))[0]
            greedy_action = np.argmax(self.model.predict(x=np.matrix(state))[0])
            action_dist = [1 if each == greedy_action else 0 for each in self.actions]
            return action_dist

        return policy_fn
