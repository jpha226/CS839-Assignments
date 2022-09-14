# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import numpy as np
from keras import backend as K
from keras.layers import Dense, Softmax, Input
from keras.optimizers import Adam
from keras.models import Model


from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


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

            Use:
                K.log: Element-wise log.
                K.sum: Sum of a tensor.
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        raise NotImplementedError

    return loss


def critic_loss():
    def loss(advantage, predicted_output):
        """
            The integral of the critic gradient

            args:
                advantage: advantage of each action a_t (one-hot encoded).
                predicted_output: Predicted state value.

            Use:
                K.sum: Sum of a tensor.
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        raise NotImplementedError

    return loss


class A2C(AbstractSolver):

    def __init__(self, env, options):
        super().__init__(env, options)
        self.actor_critic = self.build_actor_critic()
        self.policy = self.create_greedy_policy()

    def build_actor_critic(self):
        layers = self.options.layers

        states = Input(shape=self.env.observation_space.shape)
        z = states
        for l in layers[:-1]:
            z = Dense(l, activation='relu')(z)

        # Actor and critic have a seperated final fully connected layer
        z_a = Dense(layers[-1], activation='tanh')(z)
        z_a = Dense(self.env.action_space.n, activation='tanh')(z_a)
        z_c = Dense(layers[-1], activation='relu')(z)

        probs = Softmax(name='actor_output')(z_a)
        value = Dense(1, activation='linear', name='critic_output')(z_c)

        model = Model(inputs=[states], outputs=[probs, value])
        model.compile(optimizer=Adam(lr=self.options.alpha),
                      loss={'actor_output': actor_loss(),
                            'critic_output': critic_loss()},
                      loss_weights={'actor_output': 1.0, 'critic_output': 1.0})

        return model

    def create_greedy_policy(self):
        """
        Creates a greedy policy.


        Returns:
            A function that takes an observation as input and returns a greedy
            action.
        """

        def policy_fn(state):
            return np.argmax(self.actor_critic.predict([[state]])[0][0])

        return policy_fn

    def train_episode(self):
        """
        Run a single episode of the A2C algorithm

        Use:
            self.actor_critic: actor-critic network that is being learned.
            self.policy(state): Returns action probabilities.
            self.options.steps: Maximal number of steps per episode.
            np.random.choice(len(probs), probs): Randomly select an element
                from probs (a list) based on the probability distribution in probs.
            self.step(action): Performs an action in the env.
            np.zeros(): Return an array of zeros with the a given shape.
            self.env.reset(): Resets the env.
            self.options.gamma: Gamma discount factor.
            self.actor_critic.fit(): Train the policy network at the end of an episode on the
                observed transitions for exactly 1 epoch.
            self.actor_critic.predict([[state]])[1][0]: the predicted state value for 'state'
        """

        state = self.env.reset()
        states=[]
        actions = []
        deltas=[]
        for _ in range(self.options.steps):
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################

        # One-hot encoding for actions
        actions_one_hot = np.zeros([len(actions), self.env.action_space.n])
        actions_one_hot[np.arange(len(actions)), actions] = 1

        deltas = np.array(deltas)

        # Update actor critic
        self.actor_critic.fit(x=[np.array(states)],
                              y={'actor_output': deltas * actions_one_hot, 'critic_output': deltas},
                              epochs=1, batch_size=self.options.batch_size, verbose=0)

    def __str__(self):
        return "A2C"

    def plot(self, stats):
        plotting.plot_episode_stats(stats)
