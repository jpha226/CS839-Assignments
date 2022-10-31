# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import gym
import optparse
import sys
import os
import random
import numpy as np

if "../" not in sys.path:
  sys.path.append("../")

from lib import plotting
from Solvers.Abstract_Solver import AbstractSolver, Statistics
import Solvers.Available_solvers as avs
from lib.envs.gridworld import GridworldEnv
from lib.envs.blackjack import BlackjackEnv
from lib.envs.cliff_walking import CliffWalkingEnv
from lib.envs.windy_gridworld import WindyGridworldEnv


def build_parser():
    parser = optparse.OptionParser(description='Run a specified RL algorithm on a specified domain.')
    parser.add_option("-s", "--solver", dest="solver", type="string", default="random",
                      help='Solver from ' + str(avs.solvers))
    parser.add_option("-d", "--domain", dest="domain", type="string", default="Gridworld",
                      help='Domain from OpenAI Gym')
    parser.add_option("-o", "--outfile", dest="outfile", default="out",
                      help="Write results to FILE", metavar="FILE")
    parser.add_option("-x", "--experiment_dir", dest="experiment_dir", default="Experiment",
                      help="Directory to save Tensorflow summaries in", metavar="FILE")
    parser.add_option("-e", "--episodes", type="int", dest="episodes", default=500,
                      help='Number of episodes for training')
    parser.add_option("-t", "--steps", type="int", dest="steps", default=10000,
                      help='Maximal number of steps per episode')
    parser.add_option("-l", "--layers", dest="layers", type="string", default="[24,24]",
                      help='size of hidden layers in a Deep neural net. e.g., "[10,15]" creates a net where the'
                           'Input layer is connected to a layer of size 10 that is connected to a layer of size 15'
                           ' that is connected to the output')
    parser.add_option("-a", "--alpha", dest="alpha", type="float", default=0.5,
                      help='The learning rate (alpha) for updating state/action values')
    parser.add_option("-r", "--seed", type="int", dest="seed", default=random.randint(0, 9999999999),
                      help='Seed integer for random stream')
    parser.add_option("-G", "--graphics", type="int", dest="graphics_every", default=0,
                      help='Graphic rendering every i episodes. i=0 will present only one, post training episode.'
                           'i=-1 will turn off graphics. i=1 will present all episodes.')
    parser.add_option("-g", "--gamma", dest="gamma", type="float", default=1.00,
                      help='The discount factor (gamma)')
    parser.add_option("-p", "--epsilon", dest="epsilon", type="float", default=0.1,
                      help='Initial epsilon for epsilon greedy policies (might decay over time)')
    parser.add_option("-P", "--final_epsilon", dest="epsilon_end", type="float", default=0.1,
                      help='The final minimum value of epsilon after decaying is done')
    parser.add_option("-c", "--decay", dest="epsilon_decay", type="float", default=0.99,
                                        help='Epsilon decay factor')
    parser.add_option("-m", "--replay", type="int", dest="replay_memory_size", default=500000,
                      help='Size of the replay memory')
    parser.add_option("-N", "--update", type="int", dest="update_target_estimator_every", default=10000,
                      help='Copy parameters from the Q estimator to the target estimator every N steps.')
    parser.add_option("-b", "--batch_size", type="int", dest="batch_size", default=32,
                      help='Size of batches to sample from the replay memory')
    parser.add_option("-n", "--n_step", type="int", dest="n", default=1,
                      help='Value of n for n-step returns')
    parser.add_option('--no-plots', help='Option to disable plots if the solver results any',
            dest = 'disable_plots', default = False, action = 'store_true')
    return parser

def readCommand(argv):
    parser = build_parser()
    (options, args) = parser.parse_args(argv)
    return options

def getEnv(domain):
    if domain == "Blackjack":
        return BlackjackEnv()
    elif domain == "Gridworld":
        return GridworldEnv()
    elif domain == "CliffWalking":
        return CliffWalkingEnv()
    elif domain == "WindyGridworld":
        return WindyGridworldEnv()
    else:
        try:
            return gym.make(domain)
        except:
            assert False, "Domain must be a valid (and installed) Gym environment"


def parse_list(string):
    string.strip()
    string = string[1:-1].split(',') # Change "[0,1,2,3]" to '0', '1', '2', '3'
    l = []
    for n in string:
        l.append(int(n))
    return l

def main(options):
    resultdir = "Results/"
    resultdir = os.path.abspath("./{}".format(resultdir))
    options.experiment_dir = os.path.abspath("./{}".format(options.experiment_dir))

    # Create result file if one doesn't exist
    print(os.path.join(resultdir, options.outfile + '.csv'))
    if not os.path.exists(os.path.join(resultdir, options.outfile + '.csv')):
        with open(os.path.join(resultdir, options.outfile + '.csv'), 'w+') as result_file:
            result_file.write(AbstractSolver.get_out_header())

    random.seed(options.seed)
    env = getEnv(options.domain)
    env._max_episode_steps = options.steps
    print("Domain state space is {}".format(env.observation_space))
    print("Domain action space is {}".format(env.action_space))
    try:
        options.layers = parse_list(options.layers)
    except ValueError:
        raise Exception('layers argument doesnt follow int array conventions i.e., [<int>,<int>,<int>,...]')
    except:
        pass
    solver = avs.get_solver_class(options.solver)(env,options)

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(options.episodes),
        episode_rewards=np.zeros(options.episodes))

    with open(os.path.join(resultdir, options.outfile + '.csv'), 'a+') as result_file:
        result_file.write('\n')
        for i_episode in range(options.episodes):
            if options.graphics_every > 0 and i_episode % options.graphics_every == 0:
                solver.render = True
            solver.init_stats()
            solver.statistics[Statistics.Episode.value] += 1
            env.reset()
            solver.train_episode()
            solver.render = False
            result_file.write(solver.get_stat() + '\n')
            # Decay epsilon
            if options.epsilon > options.epsilon_end:
                options.epsilon *= options.epsilon_decay
            # Update statistics
            stats.episode_rewards[i_episode] = solver.statistics[Statistics.Rewards.value]
            stats.episode_lengths[i_episode] = solver.statistics[Statistics.Steps.value]
            print("Episode {}: Reward {}, Steps {}".format(i_episode+1,solver.statistics[Statistics.Rewards.value],
                                                           solver.statistics[Statistics.Steps.value]))
        if options.graphics_every > -1:
            solver.render = True
            solver.run_greedy()

        if not options.disable_plots:
            solver.plot(stats)
        solver.close()
        return {'stats':stats, 'solver': solver}

if __name__ == "__main__":
    options = readCommand(sys.argv)
    main(options)
