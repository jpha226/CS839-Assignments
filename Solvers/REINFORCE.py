# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Solvers.Abstract_Solver import AbstractSolver, Statistics
from lib import plotting


class ActorCritic(nn.Module):
    """Actor and Critic Networks for REINFORCE and A2C."""

    def __init__(self, state_size, action_size, layers=[]):
        """
        state_size: int dimensionality of state space.
        action_size: number of actions in each state.
        layers: list of integers specifying number of hidden units in each
        neural network layer.
        """
        super().__init__()
        torch.manual_seed(0)
        actor_layers = []
        critic_layers = []
        in_size = state_size
        for layer_size in layers:
            actor_layers.append(nn.Linear(in_size, layer_size))
            actor_layers.append(nn.Tanh())
            critic_layers.append(nn.Linear(in_size, layer_size))
            critic_layers.append(nn.Tanh())
            in_size = layer_size

        actor_layers += [nn.Linear(in_size, action_size), nn.Softmax()]
        self.actor_critic_net = nn.Sequential(
            *actor_layers
        )

        critic_layers += [nn.Linear(in_size, 1)]
        self.critic_net = nn.Sequential(
            *critic_layers
        )

    def action_probs(self, state):
        """
        Output action probabilities for given state input.


        state: a torch.tensor of dimensionality [None, state_size]
        """
        return self.actor_critic_net(state)

    def log_probs(self, state):
        """
        Output log action probabilities for given state input.

        state: a torch.tensor of dimensionality [None, state_size]
        """
        return torch.log(self.actor_critic_net(state))

    def value(self, state):
        """
        Predict state value for state input.

        state: a torch.tensor of dimensionality [None, state_size]
        """
        return self.critic_net(state)


class Reinforce(AbstractSolver):

    def __init__(self, env, options):
        super().__init__(env, options)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.trajectory = []
        self.actor_critic = ActorCritic(self.state_size, self.action_size,
                                        layers=options.layers)
        self.policy = self.create_greedy_policy()
        self.optimizer = optim.SGD(self.actor_critic.parameters(),
                                   lr=self.options.alpha)

    def create_greedy_policy(self):
        """
        Creates a greedy policy.
        Returns:
            A function that takes an observation as input and returns a greedy
            action.
        """

        def policy_fn(state):

            return np.argmax(
                self.actor_critic.action_probs(
                    torch.tensor(state, dtype=torch.float32)).detach().numpy()
                )

        return policy_fn

    def train_episode(self):
        """
        Run a single episode of the REINFORCE algorithm with approximated state
        value as baseline

        Some functions that may be useful:
            self.actor_critic: Policy / value network that is being learned.
            self.actor_critic.action_probs(state): Returns action probabilities
                for a given torch.tensor state.
            self.options.steps: Maximal number of steps per episode.
            np.random.choice(len(probs), probs): Randomly select an element
                from probs (a list) based on the probability distribution in
                probs.
            self.step(action): Performs an action in the env.
            np.zeros(): Return an array of zeros with the a given shape.
            self.env.reset(): Resets the env.
            self.options.gamma: Gamma discount factor.
            self.actor_critic_baseline.fit(): Train the policy network at the
                end of an episode on the
                observed transitions for exactly 1 epoch. Make sure that
                the cumulative rewards are discounted.
            self.actor_critic.value(states): the predicted baseline values for
                all states in 'states'
            torch.tensor(array, dtype=torch.float32): convert a python list or
                numpy array to a torch tensor.

            Note: inputs to self.actor_critic functions should be torch.tensors
        """

        # Generate a single episode by following the current policy.
        # Store states, actions, and rewards for an update at the end.
        state = self.env.reset()
        states = []
        actions = []
        rewards = []
        for _ in range(self.options.steps):
            probs = self.actor_critic.action_probs(
                torch.tensor(state, dtype=torch.float32)
            ).detach().numpy()
            action = np.random.choice(len(probs), p=probs)

            next_state, reward, done, _ = self.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

            if done:
                break

        # Compute and store returns in G
        G = np.zeros_like(rewards)
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        # TODO: compute returns and store as tensor
        returns = torch.tensor(G, dtype=torch.float32)

        # One-hot encoding for actions
        actions_one_hot = np.zeros([len(actions), self.env.action_space.n])
        actions_one_hot[np.arange(len(actions)), actions] = 1

        states = torch.tensor(states, dtype=torch.float32)
        actions_one_hot = torch.tensor(actions_one_hot)

        # Update actor and state value estimator
        self.train(states, actions_one_hot, returns)

    def train(self, states, actions_one_hot, returns):
        """
        Update policy and value function networks.
        Use policy gradient surrogate loss of return * log pi(a|s) to train
        policy network.
        Use MSE loss to train value function.
        """
        predictions = self.actor_critic.value(states)
        log_probs = torch.sum(
            self.actor_critic.log_probs(states) * actions_one_hot, axis=-1
            )

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        # Compute advantages
        deltas = [[0]]  # TODO: set the REINFORCE advantage: (G - baselines)

        # TODO: Compute policy and critic loss at each state-action-return
        # tuple passed in.
        policy_loss = 0.0
        critic_loss = 0.0

        loss = policy_loss.mean() + critic_loss.mean()
        loss.backward()
        self.optimizer.step()

    def __str__(self):
        return "REINFORCE"

    def plot(self, stats):
        plotting.plot_episode_stats(stats)
