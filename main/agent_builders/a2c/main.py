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
            The policy gradient loss function.
            Note that you are required to define the Loss^PG
            which should be the integral of the policy gradient
            The "returns" is the one-hot encoded (return - baseline) value for each action a_t
            ('0' for unchosen actions).

            args:
                advantage: advantage of each action a_t (one-hot encoded).
                predicted_output: Predicted actions (action probabilities).

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