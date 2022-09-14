# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The autograder for the assignments was developed by Sumedh Pendurkar (sumedhpendurkar@tamu.edu).



import unittest
import shlex
from run import main, build_parser
from copy import deepcopy
import numpy as np
import pandas as pd

def run_main(command_str):
    parser = build_parser()
    options, args = parser.parse_args(shlex.split(command_str))
    results = main(options)
    return results

def l2_distance_bounded(v1, v2, bound):
    distance = np.mean((v1 - v2) ** 2)
    return True if distance < bound else False

class vi(unittest.TestCase):
    points = 0
    @classmethod
    def setUpClass(self):
        command_str = "-s vi -d Gridworld -e 100 -g 0.9"
        self.results = run_main(command_str)

    def set_test_v(self):
        solver = self.results['solver']
        v = np.array([1.2086319679296675, 1.3206763103045178, 0.46923107590723945, 0.8616088480733963, 1.8455138935253883, 2.079230999993757, 1.9830750667821562, 1.1425017272963585, 1.3410148962839679, 2.7341845187723646, 0.10915686708240568, 2.0887591555244995, 0.12398618237655956, 2.7830735037201877, 1.8415810534076993, 0.8549790393954977])
        solver.V = v

    def test_train_episode(self):
        solver = self.results['solver']
        self.set_test_v()
        solver.train_episode()
        updated_v = solver.V
        expected_v = np.array([1.0877687711367008, 0.8713078999943813, 0.7847675601039406, 0.02825155456672257, 0.8713078999943813, 1.4607660668951281, 0.3146894602056154, 0.8798832399720495, 1.4607660668951281, 1.5047661533481689, 0.8798832399720495, 0.8798832399720495, 1.5047661533481689, 1.5047661533481689, 0.6574229480669294, 0.769481135455948])
        self.assertTrue(l2_distance_bounded(updated_v, expected_v, 1e-2), "`train_episode' function failed to provide correct output")
        self.__class__.points += 4

    def test_create_greedy_policy(self):

        self.set_test_v()
        policy = self.results['solver'].create_greedy_policy()

        greedy_actions = []
        for i in range(0, 16):
            greedy_actions += [policy(i)]
        self.assertEqual(greedy_actions, [0, 2, 2, 2, 1, 2, 3, 2, 1, 2, 3, 1, 1, 2, 3, 0], "`create_greedy_policy' function failed to provide correct output")
        self.__class__.points += 3

    def test_grid_world_1_reward(self):
        episode_rewards = self.results['stats'].episode_rewards[-1]
        expected_reward = -26.24
        self.assertEqual(expected_reward, episode_rewards,
                         'got unexpected rewards for gridworld')
        self.__class__.points += 1

    def test_grid_world_2_reward(self):
        command_str = "-s vi -d Gridworld -e 10 -g 0.4"
        results = run_main(command_str)
        episode_rewards = results['stats'].episode_rewards[-1]
        expected_reward = -18.64
        self.assertEqual(expected_reward, episode_rewards,
                         'got unexpected rewards for gridworld')
        self.__class__.points += 1

    def test_frozen_lake_reward(self):
        command_str = "-s vi -d FrozenLake-v0 -e 70 -g 0.9"
        results = run_main(command_str)
        episode_rewards = results['stats'].episode_rewards[-1]
        expected_reward = 2.176
        self.assertTrue(expected_reward < episode_rewards,
                         'got unexpected rewards for gridworld')
        self.__class__.points += 1

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))

class avi(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s avi -d Gridworld -e 100 -g 0.5"
        self.results = run_main(command_str)

    def set_test_v(self):
        solver = self.results['solver']
        v = np.array([1.2086319679296675, 1.3206763103045178, 0.46923107590723945, 0.8616088480733963, 1.8455138935253883, 2.079230999993757, 1.9830750667821562, 1.1425017272963585, 1.3410148962839679, 2.7341845187723646, 0.10915686708240568, 2.0887591555244995, 0.12398618237655956, 2.7830735037201877, 1.8415810534076993, 0.8549790393954977])
        solver.V = v

    def test_train_episode_1(self):
        solver = self.results['solver']
        self.set_test_v()
        solver.train_episode()
        updated_v = solver.V
        expected_v = np.array([1.2086319679296675, 1.3206763103045178, 0.46923107590723945, 0.8616088480733963, 1.8455138935253883, 2.079230999993757, 1.9830750667821562, 1.1425017272963585, 1.3410148962839679, 0.39153675186009385, 0.10915686708240568, 2.0887591555244995, 0.12398618237655956, 2.7830735037201877, 1.8415810534076993, 0.8549790393954977])
        self.assertTrue(l2_distance_bounded(updated_v, expected_v, 1e-2), "`train_episode' function returned unexpected outputs")
        self.__class__.points += 3

    def test_train_episode_2(self):
        solver = self.results['solver']
        self.set_test_v()
        for i in range(10):
            solver.train_episode()
        updated_v = solver.V
        expected_v = np.array([1.2086319679296675, -0.3396618448477411, 0.46923107590723945, 0.8616088480733963, -0.07724305323730585, -0.008462466608921915, -0.42874913635182077, 0.04437957776224977, -0.32949255185801607, 0.03961549999687852, 0.10915686708240568, 2.0887591555244995, 0.12398618237655956, -0.8042316240699531, -0.07920947329615036, 0.8549790393954977])
        self.assertTrue(l2_distance_bounded(updated_v, expected_v, 1e-2), "`train_episode' function return unexpected outputs")
        self.__class__.points += 4


    def test_grid_world_reward(self):
        episode_rewards = self.results['stats'].episode_rewards[-1]
        expected_reward = -20
        self.assertEqual(expected_reward, episode_rewards,
                         'got unexpected rewards for grid world')
        self.__class__.points += 1

    def test_frozen_lake_1_reward(self):
        command_str = "-s avi -d FrozenLake-v0 -e 60 -g 0.5"
        results = run_main(command_str)
        episode_rewards = results['stats'].episode_rewards[-1]
        expected_reward = 0.637
        self.assertTrue(expected_reward < episode_rewards,
                         'got unexpected rewards for frozen lake')
        self.__class__.points += 1

    def test_frozen_lake_2_reward(self):
        command_str = "-s avi -d FrozenLake-v0 -e 100 -g 0.7"
        results = run_main(command_str)
        episode_rewards = results['stats'].episode_rewards[-1]
        expected_reward = 0.978
        self.assertTrue(expected_reward < episode_rewards,
                         'got unexpected rewards for frozen lake')
        self.__class__.points += 1

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))

class pi(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s pi -d Gridworld -e 100 -g 0.9"
        self.results = run_main(command_str)

    def test_policy_eval(self):
        solver = self.results['solver']
        policy = np.eye(16, k = 1)
        policy[-1][0] = 1
        solver.policy_eval()
        v = solver.V
        expected_v = np.array([0, -1., -1.9, -2.71, -1, -1.9, -2.71, -1.9, -1.9, -2.71, -1.9, -1.0, -2.71, -1.9, -1.0, 0.0])
        self.assertTrue(l2_distance_bounded(v, expected_v, 1e-3), "`policy_eval' function resulted in unexpected V")
        self.__class__.points += 3

    def test_train_episode(self):
        def dummy_policy_eval():
            pass
        solver = self.results['solver']
        solver.V = np.array([0.1, -1., -1.9, -2.71, -1.1, -1.91, -2.72, -1.92, -1.93, -2.73, -1.95, -1.2, -2.74, -1.96, -1.3, 0.])
        solver.policy_eval = dummy_policy_eval
        policy = np.argmax(solver.policy, axis = 1).tolist()
        expected_policy = [0,3,3,2,0,0,0,2,0,0,1,2,0,1,1,0]
        self.assertEqual(policy, expected_policy, "`train_episode' function return unexpected outputs")
        self.__class__.points += 4


    def test_grid_world_1_reward(self):
        episode_rewards = self.results['stats'].episode_rewards[-1]
        expected_reward = -26.24
        self.assertEqual(expected_reward, episode_rewards,
                         'got unexpected rewards for grid world')
        self.__class__.points += 1

    def test_grid_world_2_reward(self):
        command_str = "-s pi -d Gridworld -e 10 -g 0.4"
        results = run_main(command_str)
        episode_rewards = results['stats'].episode_rewards[-1]
        expected_reward = -18.64
        self.assertEqual(expected_reward, episode_rewards,
                         'got unexpected rewards for grid world')
        self.__class__.points += 1

    def test_frozen_lake_reward(self):
        command_str = "-s pi -d FrozenLake-v0 -e 5 -g 0.5"
        results = run_main(command_str)
        episode_rewards = results['stats'].episode_rewards[-1]
        expected_reward = 0.634
        self.assertTrue(expected_reward < episode_rewards,
                         'got unexpected rewards for frozen lake')
        self.__class__.points += 1

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))

class mc(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s mc -d Blackjack -e 0 -g 0.9 -p 0.1 --no-plots"
        #command_str = "-s mc -d Blackjack -e 500000 -g 1.0 -p 0.1 --no-plots"
        self.results = run_main(command_str)

    def test_make_epsilon_greedy_policy(self):
        solver = self.results['solver']
        policy = solver.make_epsilon_greedy_policy()
        solver.Q[0][0] = 0.3
        solver.Q[0][1] = 0.1
        self.assertTrue(l2_distance_bounded(np.array([0.95, 0.05]), policy(0), 1e-8))
        self.__class__.points += 2

    def test_create_greedy_policy(self):
        solver = self.results['solver']
        policy = solver.create_greedy_policy()
        solver.Q[1][0] = 0.3
        solver.Q[1][1] = 0.1
        predict_action = policy(1)
        self.assertTrue(predict_action == 0, "`create_greedy_policy' returns unexpected policy")
        solver.Q[1][0] = 0.1
        solver.Q[1][1] = 0.3
        predict_action = policy(1)
        self.assertTrue(predict_action == 1, "`create_greedy_policy' returns unexpected policy")
        self.__class__.points += 1

    def test_train_episode(self):

        def dummy_policy(state):

            if state == (14, 10, False):
                return np.array([1,0])
            else:
                return np.array([0,1])
        def dummy_reset():
            return (14, 10, False)
        def dummy_step(action):

            if action == 0:
                return (14, 9, False), -1, False, ""
            else:
                return (23, 2, False), -1, True, ""

        solver = self.results['solver']
        solver.policy = dummy_policy
        solver.env.reset = dummy_reset
        solver.step = dummy_step
        solver.train_episode()
        self.assertEqual(list(solver.Q[(14,10,False)]), [-1.9, 0], "`train_episode' function return unexpected outputs")
        self.assertEqual(list(solver.Q[(14,9,False)]), [0, -1], "`train_episode' function return unexpected outputs")
        self.assertEqual(list(solver.Q[(23,2,False)]), [0, 0], "`train_episode' function return unexpected outputs")
        self.__class__.points += 5


    def test_blackjack_1_reward(self):
        command_str = "-s mc -d Blackjack -e 500000 -g 1.0 -p 0.1 --no-plots"
        results = run_main(command_str)
        Q_ar = np.zeros((21,21, 2, 2))
        solver = results['solver']
        for key, val in solver.Q.items():
            x, y, z = key
            z = 0 if z is False else 1
            Q_ar[x-1][y][z][0] = val[0]
            Q_ar[x-1][y][z][1] = val[1]
        expected_Q_ar = np.load("TestData/mc_rewards_mean_ar.npy")
        self.assertTrue(l2_distance_bounded(expected_Q_ar, Q_ar, 0.03),
                         'got unexpected rewards for blackjack')
        self.__class__.points += 2

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))


class mcis(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s mcis -d WindyGridworld -e 0 -g 0.3 --no-plots"
        self.results = run_main(command_str)
        solver = self.results['solver']

    def test_train_episode(self):
        solver = self.results['solver']

        def dummy_behavior_policy(state):
            action = np.ones((4)) * 0.01 / 3
            action[state] = 0.99
            return action

        def dummy_target_policy(state):
            action = np.zeros((4))
            action[state] = 1
            return action

        def dummy_reset():
            return 0

        def dummy_step(action):
            if action == 3:
                return action, 0, True, ""
            else:
                return action + 1, -1, False, ""
        solver.target_policy = dummy_target_policy
        solver.behavior_policy = dummy_behavior_policy
        solver.env.reset = dummy_reset
        solver.step = dummy_step
        solver.train_episode()
        self.assertTrue(l2_distance_bounded(solver.Q[0], np.array([-1.39, 0, 0, 0]), 1e-12), "`train_episode' function return unexpected outputs")
        self.__class__.points += 2
        self.assertTrue(l2_distance_bounded(solver.Q[1], np.array([0, -1.3, 0, 0]), 1e-12), "`train_episode' function return unexpected outputs")
        self.__class__.points += 2
        self.assertTrue(l2_distance_bounded(solver.Q[2], np.array([0, 0, -1, 0]), 1e-12), "`train_episode' function return unexpected outputs")
        self.__class__.points += 1
        self.assertTrue(l2_distance_bounded(solver.Q[3], np.array([0, 0, 0, 0]), 1e-12), "`train_episode' function return unexpected outputs")
        self.__class__.points += 1


    def test_blackjack_1_reward(self):
        command_str = "-s mcis -d Blackjack -e 500000 -g 0.6 --no-plots"
        results = run_main(command_str)
        Q_ar = np.zeros((21,21, 2, 2))
        solver = results['solver']
        for key, val in solver.Q.items():
            x, y, z = key
            z = 0 if z is False else 1
            Q_ar[x-1][y][z][0] = val[0]
            Q_ar[x-1][y][z][1] = val[1]
        expected_Q_ar = np.load("TestData/mcis_rewards_mean_ar.npy")
        self.assertTrue(l2_distance_bounded(expected_Q_ar, Q_ar, 0.03),
                         'got unexpected rewards for blackjack')
        self.__class__.points += 4

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))

class ql(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s ql -d Blackjack -e 0 -a 0.5 -g 0.3 -p 0.1 --no-plots"
        self.results = run_main(command_str)

    def test_make_epsilon_greedy_policy(self):
        solver = self.results['solver']
        policy = solver.epsilon_greedy_action
        solver.Q[0][0] = 0.3
        solver.Q[0][1] = 0.1
        np.random.seed(10)
        import random
        random.seed(10)
        #test = [(sum([policy(0) for x in range(1000)])) for y in range(100)]
        self.assertTrue(l2_distance_bounded(np.array([0.95, 0.05]),
            policy(0), 1e-8), "`make_epsilon_greedy_policy' returns unexpected policy")
        self.__class__.points += 2

    def test_create_greedy_policy(self):
        solver = self.results['solver']
        policy = solver.create_greedy_policy()
        old_Q = deepcopy(solver.Q)
        solver.Q[1][0] = 0.3
        solver.Q[1][1] = 0.1
        predict_action = policy(1)
        self.assertTrue(predict_action == 0, "`create_greedy_policy' returns unexpected policy")
        solver.Q[1][0] = 0.1
        solver.Q[1][1] = 0.3
        predict_action = policy(1)
        self.assertTrue(predict_action == 1, "`create_greedy_policy' returns unexpected policy")
        self.__class__.points += 1
        solver.Q = old_Q

    def test_train_episode(self):

        def dummy_policy(state):

            if state == (14, 10, False):
                return 0
            else:
                return 1

        def dummy_reset():
            return (14, 10, False)

        def dummy_step(action):

            if action == 0:
                return (14, 9, False), -1, False, ""
            else:
                return (23, 2, False), -1, True, ""

        solver = self.results['solver']
        solver.make_epsilong_greedy_policy = dummy_policy
        solver.env.reset = dummy_reset
        solver.step = dummy_step
        solver.train_episode()
        self.assertEqual(list(solver.Q[(14,10,False)]), [-0.5, 0], "`train_episode' function return unexpected outputs")
        self.assertEqual(list(solver.Q[(14,9,False)]), [-.5, -.5], "`train_episode' function return unexpected outputs")
        self.assertEqual(list(solver.Q[(23,2,False)]), [0, 0], "`train_episode' function return unexpected outputs")
        self.__class__.points += 5


    def test_cliff_walking_reward(self):
        command_str = "-s ql -d CliffWalking -e 500 -a 0.5 -g 1.0 -p 0.1 --no-plots"
        results = run_main(command_str)
        solver = results['solver']
        stats = results['stats']
        smoothing_window = 10
        ep_len = stats.episode_lengths
        self.assertTrue(np.mean(ep_len[:30]) > 25 and np.mean(ep_len[150:]) < 15,
                         'got unexpected rewards for cliff walking')
        self.__class__.points += 1
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

        self.assertTrue(np.max(rewards_smoothed) > -15 and np.mean(ep_len[150:]) > -40,
                         'got unexpected rewards for cliff walking')
        self.__class__.points += 1

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))

class sarsa(unittest.TestCase):
    points = 0

    @classmethod
    def setUpClass(self):
        command_str = "-s sarsa -d WindyGridworld -e 0 -a 0.5 -g 0.3 -p 0.1 --no-plots"
        self.results = run_main(command_str)

    def test_make_epsilon_greedy_policy(self):
        solver = self.results['solver']
        policy = solver.epsilon_greedy_action
        old_Q = deepcopy(solver.Q)
        solver.Q[0][0] = 0.3
        solver.Q[0][1] = 0.1
        solver.Q[0][2] = 0.2
        solver.Q[0][3] = 0.0
        self.assertTrue(l2_distance_bounded(np.array([0.925, 0.025, 0.025, 0.025]),
            policy(0), 1e-8), "`make_epsilon_greedy_policy' returns unexpected policy")
        solver.Q = old_Q
        self.__class__.points += 2

    def test_create_greedy_policy(self):
        solver = self.results['solver']
        policy = solver.create_greedy_policy()
        old_Q = deepcopy(solver.Q)
        solver.Q[1][0] = 0.3
        solver.Q[1][1] = 0.1
        solver.Q[1][2] = 0.1
        solver.Q[1][3] = 0.1
        predict_action = policy(1)
        self.assertTrue(predict_action == 0, "`create_greedy_policy' returns unexpected policy")
        solver.Q[1][0] = 0.1
        solver.Q[1][1] = 0.3
        solver.Q[1][2] = 0.1
        solver.Q[1][3] = 0.1
        predict_action = policy(1)
        self.assertTrue(predict_action == 1, "`create_greedy_policy' returns unexpected policy")
        solver.Q = old_Q
        self.__class__.points += 1

    def test_train_episode(self):

        def dummy_policy(state):
            action = np.zeros((4))
            action[state] = 1
            return action

        def dummy_reset():
            return 0

        def dummy_step(action):
            if action == 3:
                return action, 0, True, ""
            else:
                return action + 1, -1, False, ""

        solver = self.results['solver']
        solver.epsilon_greedy_action = dummy_policy
        solver.env.reset = dummy_reset
        solver.step = dummy_step
        solver.train_episode()
        solver.train_episode()
        self.assertEqual(list(solver.Q[0]), [-0.825, 0, 0, 0], "`train_episode' function return unexpected outputs")
        self.assertEqual(list(solver.Q[1]), [0, -0.825, 0, 0], "`train_episode' function return unexpected outputs")
        self.assertEqual(list(solver.Q[2]), [0, 0, -0.75, 0], "`train_episode' function return unexpected outputs")
        self.assertEqual(list(solver.Q[3]), [0, 0, 0, 0], "`train_episode' function return unexpected outputs")
        self.__class__.points += 5


    def test_cliff_walking_reward(self):
        command_str = "-s ql -d CliffWalking -e 500 -a 0.5 -g 1.0 -p 0.1 --no-plots"
        results = run_main(command_str)
        solver = results['solver']
        stats = results['stats']
        smoothing_window = 10
        ep_len = stats.episode_lengths
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        self.assertTrue(np.mean(ep_len[:10]) > 25 and np.mean(ep_len[150:]) < 15,
                         'got unexpected rewards for cliff walking')
        self.__class__.points += 1
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

        self.assertTrue(np.max(rewards_smoothed) > -15 and np.mean(ep_len[150:]) > -40,
                         'got unexpected rewards for cliff walking')
        self.__class__.points += 1

    @classmethod
    def tearDownClass(cls):
        print("\nTotal Points: {} / 10".format(cls.points))

if __name__ == '__main__':
    import sys
    assert(len(sys.argv) == 2)
    unittest.main()
