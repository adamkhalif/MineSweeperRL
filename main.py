from MineSweeperEnv import MineSweeperEnv
from dqn_model import DoubleQLearningModel, ExperienceReplay
import torch
from q_learning import train_loop_ddqn
from plot_functions import plot_reward
import json


def write_to_json(result_dict, filepath):
    with open(filepath, "w") as f:
        json.dump(result_dict, f)
    return


def load_from_json(filepath):
    with open(filepath, "r") as f:
        json_data = json.load(f)
    return json_data

if __name__ == "__main__":
    reward = [10, -3, 1, 0]  # win lose progress no progress
    filepath = "Result_FIXED_BOMBS.json"
    num_episodes = 500
    batch_size = 128
    gamma = 0.2
    learning_rate = 1e-4
    HEIGHT = 3
    WIDTH = 3
    N_BOMBS = 1
    eps = 1.
    eps_end = 0
    eps_decay = 1/(num_episodes-(num_episodes*0.2))
    conv = True
    dim = HEIGHT

    # Create the environment
    env = MineSweeperEnv(HEIGHT, WIDTH, N_BOMBS, reward)

    # Enable visualization? Does not work in all environments.
    enable_visualization = False

    # Initializations
    num_actions = env.n_actions
    num_states = env.observation_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    # Object holding our online / offline Q-Networks
    ddqn = DoubleQLearningModel(device, num_states, num_actions, learning_rate, conv=conv, dim=dim)

    # Create replay buffer, where experience in form of tuples <s,a,r,s',t>, gathered from the environment is stored
    # for training
    replay_buffer = ExperienceReplay(device, num_states)

    # Train
    R_buffer, R_avg, eps, avg_wins, i, ep_reward, avg_progress = train_loop_ddqn(ddqn, env, replay_buffer, num_episodes, enable_visualization=enable_visualization, batch_size=batch_size, gamma=gamma, eps=eps, eps_end=eps_end, eps_decay=eps_decay)

    result_dict = {}
    result_dict["epsilon"] = eps
    result_dict["avg_wins"] = avg_wins
    result_dict["episodes"] = i
    result_dict["ep_reward"] = ep_reward
    result_dict["running_average"] = R_avg
    result_dict["average_progress"] = avg_progress


    write_to_json(result_dict, filepath)
    data = load_from_json(filepath)
    plot_reward(data)
