# Acknowledgments
The course codebase and assignments are provided by Prof. Guni Sharon at Texas A&M University. They were developed as part of his course CSCE-689: Reinforcement Learning course taught at Texas A&M.

# Overview
In the programming exercises for this semester, you will be implementing various RL algorithms. More specifically, you will be completing implementations. This codebase provides the starter code for algorithms as well as code to run the algorithms.

# Installation
The framework for the assignments can be set up by installing the following packages with 'pip' using Python 3.6 (you are highly encouraged to set up a virtual env to manage dependencies). You can setup the environment by running  `virtualenv -p python3.6 cs839 .` You can change into the environment by running   `source activate cs839` You can install all the requirements with pip:  `pip install -r requirements.txt`

# Running Code
In order to execute the code, type  `python run.py`  along with the following arguments as needed. When an argument is not provided, its default value will be assumed.

  `-s <solver>`  : Specify which RL algorithm to run. E.g., "-s vi" will execute the Value Iteration algorithm. Default=random control.
  
  `-d <domain>`  : Chosen environment from OpenAI Gym. E.g., "-d Gridworld" will run the Gridworld domain. Default="Gridworld".
  
  `-o <name>`  : The result output file name. Default="out.csv".
  
  `-x <dir>`  : Directory to save Tensorflow summaries in. Default="Experiment".
  
  `-e <int>`  : Number of episodes for training. Default=500.
  
  `-t <int>`  : Maximal number of steps per episode. Default=10,000.
  
  `-l <[int,int,...,int]>`  : Structure of a Deep neural net. E.g., "[10,15]" creates a net where the Input layer is connected to a hidden layer of size 10 that is connected to a hidden layer of size 15 that is connected to the output. Default=[24,24].
  
  `-a <alpha>`  : The learning rate (alpha). Default=0.5.
  
  `-r <seed>`  : Seed integer for a random stream. Default=random value from [0, 9999999999].
  
  `-G <i>`  : Graphic rendering every i episodes. i=0 will present only a post training episode. i=-1 will turn off graphics completely. i=1 will present all episodes. Default=0.
  
  `-g <gamma>`  : The discount factor (gamma). Default=1.00.
  
  `-p <epsilon>`  : Initial epsilon for epsilon greedy policies (can set up to decay over time). Default=0.1.
  
  `-P <epsilon>`  : The minimum value of epsilon at which decaying is terminated (can be zero). Default=0.1.
  
  `-c <rate>`  : Epsilon decay rate (can be 1 = no decay). Default=0.99.
  
  `-m <size>`  : Size of the replay memory. Default=500,000.
  
  `-N <n>`  : Copy parameters from the trained estimator to the target estimator every n steps. Default=10,000.
  
  `-b <size>`  : Size of the training mini batches. Default=32.
  
  `--no-plots`;  : To avoid generating plots at the end.
