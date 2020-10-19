from MineSweeperEnv import MineSweeperEnv
from dqn_model import DoubleQLearningModel, ExperienceReplay
import torch
from q_learning import train_loop_ddqn
from plot_functions import plot_reward

# Create the environment
env = MineSweeperEnv(3,3,1)

# Enable visualization? Does not work in all environments.
enable_visualization = False

# Initializations
num_actions = env.n_actions
num_states = env.observation_space.shape[0]


num_episodes = 2000
batch_size = 128
gamma = 0.01
learning_rate = 1e-4

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# Object holding our online / offline Q-Networks
ddqn = DoubleQLearningModel(device, num_states, num_actions, learning_rate)

# Create replay buffer, where experience in form of tuples <s,a,r,s',t>, gathered from the environment is stored
# for training
replay_buffer = ExperienceReplay(device, num_states)

# Train
R, R_avg = train_loop_ddqn(ddqn, env, replay_buffer, num_episodes, enable_visualization=enable_visualization, batch_size=batch_size, gamma=gamma)

plot_reward(R,R_avg)
