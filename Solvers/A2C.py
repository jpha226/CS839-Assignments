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
from Solvers.REINFORCE import ActorCritic
from lib import plotting


class A2C(AbstractSolver):

    def __init__(self, env, options):
        super().__init__(env, options)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.trajectory = []
        self.actor_critic = ActorCritic(self.state_size, self.action_size,
                                        layers=options.layers)
        self.policy = self.create_greedy_policy()
        self.optimizer = optim.SGD(self.actor_critic.parameters(),
                                   options.alpha)

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
                    torch.tensor(state, dtype=torch.float32)
                ).detach().numpy()
            )

        return policy_fn

    def train_episode(self):
        """
        Run a single episode of the A2C algorithm

        Useful functions and objects:
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
            self.options.n: n for n-step returns.
        """

        state = self.env.reset()
        states = []
        actions = []
        rewards = []

        for t in range(self.options.steps):

            probs = self.actor_critic.action_probs(
                torch.tensor(state, dtype=torch.float32)
            ).detach().numpy()
            action = np.random.choice(len(probs), p=probs)

            next_state, reward, done, _ = self.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            if t > 0 and t % self.options.n == 0:
                self.train(states, actions, rewards, next_state, done)
                states, actions, rewards = [], [], []

            state = next_state

            if done:
                break

        if len(states) > 1:
            self.train(states, actions, rewards, next_state, done)

    def train(self, states, actions, rewards, next_state, done):
        """
        Perform single A2C update.

        states: list of states.
        actions: list of actions taken.
        rewards: list of rewards received.
        next_state: next state received after final action.
        done: if episode ended after last action was taken.
        """

        states_tensor = torch.tensor(states, dtype=torch.float32)

        # One-hot encoding for actions
        actions_one_hot = np.zeros([len(actions), self.env.action_space.n])
        actions_one_hot[np.arange(len(actions)), actions] = 1
        actions_one_hot = torch.tensor(actions_one_hot)

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        # Compute returns
        returns = np.zeros_like(rewards)
        # TODO: Compute bootstrapped returns for each state-action in states
        # and actions
        returns = torch.tensor(returns, dtype=torch.float32)

        values = self.actor_critic.value(states_tensor)

        # TODO: Compute advantages for each state-action pair in states and
        # actions.

        log_probs = torch.sum(
            self.actor_critic.log_probs(states_tensor) * actions_one_hot,
            axis=-1
        )

        # Compute actor and critic losses
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        # TODO: compute these losses.
        # Useful functions: torch.square for critic loss.
        policy_loss = 0.0
        critic_loss = 0.0
        loss = policy_loss.mean() + critic_loss.mean()
        loss.backward()
        self.optimizer.step()

    def __str__(self):
        return "A2C"

    def plot(self, stats):
        plotting.plot_episode_stats(stats)
