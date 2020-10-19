from MineSweeperEnv import MineSweeperEnv
from dqn_model import DoubleQLearningModel, ExperienceReplay
import torch
from q_learning import train_loop_ddqn
from plot_functions import plot_reward



reward = [1, -4.5, 0.5, 0] #win lose progress no progress
num_episodes = 10000
batch_size = 1024
gamma = 0.01
learning_rate = 1e-4
HEIGHT = 3
WIDTH = 3
N_BOMBS = 1
eps = 1.
eps_end = 0
eps_decay = 1/(num_episodes-200)
conv=True
dim = HEIGHT

# Create the environment
env = MineSweeperEnv(HEIGHT, WIDTH, N_BOMBS,reward)

# Enable visualization? Does not work in all environments.
enable_visualization = False

# Initializations
num_actions = env.n_actions
num_states = env.observation_space.shape[0]




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
# Object holding our online / offline Q-Networks
ddqn = DoubleQLearningModel(device, num_states, num_actions, learning_rate, conv=conv, dim=dim)

# Create replay buffer, where experience in form of tuples <s,a,r,s',t>, gathered from the environment is stored
# for training
replay_buffer = ExperienceReplay(device, num_states)

# Train
R, R_avg = train_loop_ddqn(ddqn, env, replay_buffer, num_episodes, enable_visualization=enable_visualization, batch_size=batch_size, gamma=gamma, eps=eps, eps_end=eps_end, eps_decay=eps_decay)

plot_reward(R, R_avg)
