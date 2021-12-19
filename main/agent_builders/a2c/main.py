# 
# theory pseudocode as code:
# 

# pi_theta = None
# critic_w = None
# for each_episode in episodes:
#     trajectory = []
#     state = env.reset()
#     import itertools
#     for time_index in itertools.count(0): # starting at 0
#         import random
#         action = random.choices(list_of_things, weights=pi_theta(state))[0]
#         next_state, reward, episode_is_over, _ = env.step(state, action)
#         trajectory.append((time_index, state, action, reward, next_state))
#         state = next_state
#         if episode_is_over:
#             break
#     d_theta = 0
#     d_w = 0
#     delta = defaultdict(lambda: None)
#     for time_index, state, action, reward, next_state in reversed(trajectory):
#         delta[time_index] = reward + gamma * critic_w(next_state) - critic_w(state)

import torch

def actor_loss():
    def loss(advantage, predicted_output):
        """
            args:
                advantage: advantage of each action a_t (one-hot encoded).
                predicted_output: Predicted actions (action probabilities).
            
            returns:
                integral of the policy gradient, a one-hot encoded value for each action a_t

            use:
                torch.log: Element-wise log.
                torch.sum: Sum of a tensor.
        """
        return torch.sum(advantage * -torch.log(torch.clip(predicted_output, 1e-8, 1-1e-8)))

    return loss


def critic_loss():
    def loss(advantage, predicted_output):
        """
            The integral of the critic gradient

            args:
                advantage: advantage of each action a_t (one-hot encoded).
                predicted_output: Predicted state value.

            Use:
                torch.sum: Sum of a tensor.
        """
        return torch.sum(-advantage * predicted_output)

    return loss
    


def test_runtime(number_of_episodes=1000, max_number_of_timesteps=10000):
    import itertools
    import random
    
    # 
    # create ENV
    # 
    env = gym.make("CartPole-v1") # alternate: "Breakout-v0"
    all_actions = range(env.action_space.n)
    print(f"    Observation space is {env.observation_space}")
    print(f"    Action space is {env.action_space}")
    
    # 
    # Run test
    # 
    number_of_episodes = 2000
    learning_rate = 0.001
    discount_factor = 0.99
    layers = [64,64]
    network = None # FIXME
    def build_network(layers):
        # states = Input(shape=self.state_size)
        z = states
        for l in layers[:-1]:
            z = Dense(l, activation='relu')(z)

        # Actor and critic heads have a seperated final fully connected layer
        
        # actor
        z_a = Dense(layers[-1], activation='tanh')(z)
        z_a = Dense(self.env.action_space.n, activation='tanh')(z_a)
        # critic
        z_c = Dense(layers[-1], activation='relu')(z)

        probs = Softmax(name='actor_output')(z_a)
        value = Dense(1, activation='linear', name='critic_output')(z_c)

        model = Model(inputs=[states], outputs=[probs, value])
        model.compile(
            optimizer=Adam(lr=self.options.alpha),
            loss={
                'actor_output': actor_loss(),
                'critic_output': critic_loss()
            },
            loss_weights={
                'actor_output': 1.0,
                'critic_output': 1.0
            }
        )

        return model
    def make_decision(observation):
        """
        returns:
            list of probabilities (0 to 1), one for each action in all_actions
        """
        action_probabilities = network.forward(observation)
        # weighted random choice of action
        action = random.choices(list_of_things, weights=action_probabilities)[0]
        return action 
    
    def value_approximator(observation):
        # FIXME
        return 0
    
    for each_episode_index in range(number_of_episodes):
        episode_reward = 0
        observation  = env.reset()
        # 
        # per-episode data gathering
        # 
        chosen_actions    = []
        observations      = []
        next_observations = []
        rewards           = []
        trajectory        = []
        env_stopped_the_episode = False 
        for time_index in range(max_number_of_timesteps):
            # weighted action choice
            action = make_decision(observation)
            # see what the world says
            next_observation, reward, env_stopped_the_episode, _ = env.step(action)
            episode_reward += reward
            # save info
            actions.append(action)
            observations.append(observation)
            next_observations.append(next_observation)
            next_observation, reward, env_stopped_the_episode, _ = env.step(action)
            rewards.append(reward)
            trajectory.append((time_index, observation, action, reward, next_observation, env_stopped_the_episode))
            
            observation = next_observation
            if env_stopped_the_episode:
                break
        # 
        # update critic, based on trajectory
        # 
        
        # real reward + discounted estimated-reward (multiplied by 0 if last timestep)
        observation_values = [     reward + (discount_factor*value_approximator(next_observation))     for reward, next_observation in zip(rewards, next_observations) ]
        # sometimes the last value is set to zero (env_stopped_the_episode is false when max_number_of_timesteps is hit)
        observation_values[-1] = 0 if env_stopped_the_episode else observation_values[-1]
        # difference between the partially-observed value and the pure-estimate value
        deltas = np.array([    each_observation_value - value_approximator(each_observation)    for each_observation_value, each_observation zip(observation_values, observations) ])
        
        advantage = reward + ((1.0 - done) * discount_factor * value_approximator(next_state))
                    -
                    value_approximator(state)
        
        # one-hot encoding for actions (matrix)
        actions_one_hot = np.zeros([len(actions), self.env.action_space.n])
        actions_one_hot[np.arange(len(actions)), actions] = 1
        # update network
        self.actor_critic.fit(
            x=[np.array(observations)],
            y={'actor_output': deltas * actions_one_hot, 'critic_output': deltas},
            epochs=1,
            batch_size=self.options.batch_size,
            verbose=0
        )

if __name__ == "__main__":
    test_runtime()