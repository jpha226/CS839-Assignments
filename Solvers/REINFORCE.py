# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import numpy as np
from keras import backend as K
from keras import losses
from keras.layers import Dense, Softmax, Input
from keras.optimizers import Adam
from keras.models import Model
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting


def pg_loss():
    def loss(returns, predicted_output):
        """
        The policy gradient loss function.
        Note that you are required to define the Loss^PG
        which should be the integral of the policy gradient
        The "returns" is the one-hot encoded (return - baseline) value for each action a_t
        ('0' for unchosen actions).

        args:
            returns: sum of discounted returns following each action a_t (one-hot encoded).
            predicted_output: Predicted actions (action probabilities).

        Use:
            K.log: Element-wise log.
            K.sum: sum of a tensor.
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        raise NotImplementedError

    return loss


class Reinforce(AbstractSolver):

    def __init__(self, env, options):
        super().__init__(env, options)
        self.state_size = self.env.observation_space.shape
        self.action_size = self.env.action_space.n
        self.trajectory = []
        self.actor_baseline = self.build_actor_baseline()
        self.policy = self.create_greedy_policy()

    def build_actor_baseline(self):
        layers = self.options.layers

        states = Input(shape=self.state_size)
        z = states
        for l in layers[:-1]:
            z = Dense(l, activation='relu')(z)

        # actor and critic heads have a seperated final fully connected layer
        z_a = Dense(layers[-1], activation='tanh')(z)
        z_a = Dense(self.env.action_space.n, activation='tanh')(z_a)
        z_b = Dense(layers[-1], activation='relu')(z)

        probs = Softmax(name='actor_output')(z_a)
        baseline = Dense(1, activation='linear', name='baseline_output')(z_b)

        model = Model(inputs=[states], outputs=[probs, baseline])
        model.compile(optimizer=Adam(lr=self.options.alpha),
                      loss={'actor_output': pg_loss(),
                            'baseline_output': losses.MeanSquaredError()},
                      loss_weights={'actor_output': 1.0, 'baseline_output': 1.0})

        return model

    def create_greedy_policy(self):
        """
        Creates a greedy policy.
        Returns:
            A function that takes an observation as input and returns a greedy
            action.
        """

        def policy_fn(state):
            return np.argmax(self.actor_baseline.predict([[state]])[0][0])

        return policy_fn

    def train_episode(self):
        """
        Run a single episode of the REINFORCE algorithm + approximated state value as baseline
        Use:
            self.actor_baseline: Policy network that is being learned.
            self.policy(state): Returns action probabilities.
            self.options.steps: Maximal number of steps per episode.
            np.random.choice(len(probs), probs): Randomly select an element
                from probs (a list) based on the probability distribution in probs.
            self.step(action): Performs an action in the env.
            np.zeros(): Return an array of zeros with the a given shape.
            self.env.reset(): Resets the env.
            self.options.gamma: Gamma discount factor.
            self.actor_baseline.fit(): Train the policy network at the end of an episode on the
                observed transitions for exactly 1 epoch. Make sure that
                the cumulative rewards are discounted.
            np.reshape(returns,(-1,1)): returns a reshaped np.array of size len(returns)
                where each entry is an array with a single element from the original array.
                e.g., [1,2,3,4] becomes [[1],[2],[3],[4]].
            self.actor_baseline.predict(np.array(states))[1]: the predicted baseline values for all states in 'states'
        """
        state = self.env.reset()
        states = []
        actions = []
        rewards = []
        for _ in range(self.options.steps):
            probs = self.actor_baseline.predict([[state]])[0][0]
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

        # One-hot encoding for actions
        actions_one_hot = np.zeros([len(actions), self.env.action_space.n])
        actions_one_hot[np.arange(len(actions)), actions] = 1

        # Compute one-hot encoded deltas
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        deltas = [[0]] # Set the right deltas: (returns - baselines) times the one hot vector

        # Update actor and state estimator
        self.actor_baseline.fit(x=[np.array(states)],
                              y={'actor_output': deltas, 'baseline_output': returns},
                              epochs=1, batch_size=self.options.batch_size, verbose=0)

    def __str__(self):
        return "REINFORCE"

    def plot(self, stats):
        plotting.plot_episode_stats(stats)
