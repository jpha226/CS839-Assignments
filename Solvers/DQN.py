# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import os
import random
from collections import deque
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import huber_loss
import numpy as np
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


class DQN(AbstractSolver):
    def __init__(self,env,options):
        assert (str(env.action_space).startswith('Discrete') or
        str(env.action_space).startswith('Tuple(Discrete')), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.memory = deque(maxlen=options.replay_memory_size)

    def _build_model(self):
        layers = self.options.layers

        states = Input(shape=self.env.observation_space.shape)
        z = states
        for l in layers:
            z = Dense(l, activation='relu')(z)

        q = Dense(self.env.action_space.n, activation='linear')(z)

        model = Model(inputs=[states], outputs=[q])
        model.compile(optimizer=Adam(lr=self.options.alpha), loss=huber_loss)

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def epsilon_greedy(self, state):
        """
        Apply an epsilon-greedy policy based on the given Q-function approximator and epsilon.

        Returns:
            An epsilon greedy action (as an int) for 'state'

        Use:
            self.env.action_space.n: number of avilable actions
            q_values = self.model.predict([[state]])[0]: Predicted Q values for
                'state' as a vector. One value per action.
            np.argmax(q_values): returns the action coresponding to the highest
                q value
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        return 0

    def replay(self):
        """
        TD learning for q values on past transitions

        Use:
            np.random.choice(len(probs), probs): Randomly select an element
                from probs (a list) based on the probability distribution in probs.
            self.model.predict([[state]])[0]: predicted q values as an array with entry
                per action
        """
        if len(self.memory) > self.options.batch_size:
            minibatch = random.sample(self.memory, self.options.batch_size)
            states = []
            target_q = []
            for state, action, reward, next_state, done in minibatch:
                states.append(state)
                ################################
                #   YOUR IMPLEMENTATION HERE   #
                #  Compute the target value    #
                ################################
                target_q.append(q)

            states = np.array(states)
            target_q = np.array(target_q)
            self.model.fit(states, target_q, epochs=1, verbose=0)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_episode(self):
        """
        Perform a single episode of the Q-Learning algorithm for off-policy TD
        control using a DNN Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Use:
            self.epsilon_greedy(state): return an epsilon greedy action
            self.step(action): advance one step in the environment
            self.memorize(state, action, reward, next_state, done): store the transition in the reply buffer
            self.update_target_model(): copy weights from model to target_model
            self.replay(): TD learning for q values on past transitions
            self.options.update_target_estimator_every: Copy parameters from the Q estimator to the
                target estimator every N steps
        """

        # Reset the environment
        state = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################


    def __str__(self):
        return "DQN"

    def plot(self,stats):
        plotting.plot_episode_stats(stats)

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.


        Returns:
            A function that takes an observation as input and returns a greedy
            action
        """

        def policy_fn(state):
            q_values = self.model.predict([[state]])
            return np.argmax(q_values[0])

        return policy_fn
