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

if __name__ == '__main__':
    reward = [1, -1, 0.3, 0]  # win lose progress no progress
    filepath = "Result_RANDOM_BOMBS_no_lose_first.json"
    num_episodes = 10000
    batch_size = 256
    gamma = 0
    learning_rate = 1e-4
    HEIGHT = 3
    WIDTH = 3
    N_BOMBS = 1
    eps = 1.
    eps_end = 0.1
    eps_decay = eps/(num_episodes-num_episodes*0.1)
    conv = False
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

    eps_list = []
    avg_wins_list = []
    R_avg_list = []
    wins_list = []

    R_buffer, R_avg, eps, avg_wins, i, ep_reward, R_avg_progress, wins = train_loop_ddqn(ddqn, env, replay_buffer, num_episodes, enable_visualization=enable_visualization, batch_size=batch_size, gamma=gamma, eps=eps, eps_end=eps_end, eps_decay=eps_decay)

    eps_list.extend(eps)
    avg_wins_list.extend(avg_wins)
    R_avg_list.extend(R_avg)
    wins_list.extend(wins)

    while (True):
        inp = input("Continue?: ")
        inp = int(inp)
        if inp == 1:
            eps = input("Epsilon?: ")
            eps = float(eps)
            eps_end = input("Epsilon end?: ")
            eps_end = float(eps_end)
            R_buffer, R_avg, eps, avg_wins, i, ep_reward, R_avg_progress, wins = train_loop_ddqn(ddqn, env, replay_buffer,
                                                                                           num_episodes,
                                                                                           enable_visualization=enable_visualization,
                                                                                           batch_size=batch_size,
                                                                                           gamma=gamma, eps=eps,
                                                                                           eps_end=eps_end,
                                                                                           eps_decay=eps_decay)
            eps_list.extend(eps)
            avg_wins_list.extend(avg_wins)
            R_avg_list.extend(R_avg)
            wins_list.extend(wins)
        else:
            break

    #TODO compute sliding average wins

    result_dict = {}
    result_dict["conv"] = conv
    result_dict["rewards"] = reward
    result_dict["gamma"] = gamma
    result_dict["lr"] = learning_rate
    result_dict["batch_size"] = batch_size
    result_dict["eps_decay"] = eps_decay

    result_dict["epsilon"] = eps_list
    result_dict["avg_wins"] = avg_wins_list
    result_dict["running_average"] = R_avg_list
    result_dict["wins"] = wins_list

    write_to_json(result_dict, filepath)
    data = load_from_json(filepath)
    plot_reward(data)
