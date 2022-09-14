# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

from abc import ABC, abstractmethod
import numpy as np
from enum import Enum


class AbstractSolver(ABC):

    def __init__(self,env,options):
        self.statistics = [0] * len(Statistics)
        self.env = env
        self.options = options
        self.total_steps = 0
        self.render = False

    def init_stats(self):
        self.statistics[1:] = [0] * (len(Statistics)-1)

    def step(self,action):
        """
        Take one step in the environment while keeping track of statistical information
        Param:
            action:
        Return:
            next_state: The next state
            reward: Immediate reward
            done: Is next_state terminal
            info: Gym transition information
        """
        next_state, reward, done, info = self.env.step(action)
        reward += self.calc_reward(next_state)

        # Update statistics
        self.statistics[Statistics.Rewards.value] += reward
        self.statistics[Statistics.Steps.value] += 1
        self.total_steps += 1
        if self.render:
            try:
                self.env.render()
            except:
                pass
        return next_state, reward, done, info

    def calc_reward(self,state):
        # Create a new reward function for the CartPole domain that takes into account the degree of the pole
        try:
            domain = self.env.unwrapped.spec.id
        except:
            domain = self.env.name
        if domain == 'CartPole-v1':
            x, x_dot, theta, theta_dot = state
            r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8
            r2 = (self.env.theta_threshold_radians - abs(theta)) / self.env.theta_threshold_radians - 0.5
            return r1 + r2
        return 0

    def run_greedy(self):
        """
        Run the greedy policy post learning
        """
        policy = self.create_greedy_policy()
        # Reset the environment and pick the first action
        state = self.env.reset()
        rewards = 0
        time = 0
        # One step in the environment
        for time in range(self.options.steps):
            action = policy(state)
            state, reward, done, _ = self.step(action)
            reward += reward
            if done:
                break
        return rewards, time

    def close(self):
        pass

    @abstractmethod
    def train_episode(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def create_greedy_policy(self):
        pass

    @staticmethod
    def get_out_header():
        ans = "Domain,Solver"
        for s in Statistics:
            ans += ","+s.name
        return ans

    def plot(self,stats):
        pass

    def get_stat(self):
        try:
            domain = self.env.unwrapped.spec.id
        except:
            domain = self.env.name
        ans = '{},{}'.format(domain, str(self))
        for s in Statistics:
            ans += ',' + str(self.statistics[s.value])
        return ans


class Statistics(Enum):
    Episode = 0
    Rewards = 1
    Steps = 2
