from MineSweeperEnv import MineSweeperEnv
from dqn_model import DoubleQLearningModel, ExperienceReplay
import torch
from q_learning import train_loop_ddqn
from plot_functions import plot_reward
import json

"""
Run main.py to start a training period. When the chosen episodes have finished the user can chose to either:
    - Chose to continue the training: 'Continue?:' yes if input = 1
        -   Chose a new epsilon: 'Epsilon?:'  
        -   Chose a new epsilon: 'Epsilon end?:'
        -   Chose number of episodes: 'Number of episodes?:' 
    - Terminate the training and save and plot the result (by choosing some other int not = 1)
    
    Some of the code have been copied from HA3, i.e. train_loop_ddqn() and creation of ddqn.
"""


def compute_moving_average(data):
    """
    A method that computes the moving average given a sliding window.
    Found on kite.com:
    https://www.kite.com/python/answers/how-to-find-the-moving-average-of-a-list-in-python
    Args:
        data: list with bools containing information if the agent won (1) or lost (0).

    Returns:
        moving average of the given input and sliding window size
    """
    window_size = 100
    i = 0
    moving_averages = []
    while i < len(data) - window_size + 1:
        this_window = data[i: i + window_size]

        window_average = sum(this_window) / window_size*100
        moving_averages.append(window_average)
        i += 1
    return moving_averages


def write_to_json(result_dict, filepath):
    """
    Function for writing to .json-file
    Args:
        result_dict: dictionary containing results
        filepath: filepath to file that is to be created

    Returns:
        -
    """
    with open(filepath, "w") as f:
        json.dump(result_dict, f)
    return


def load_from_json(filepath):
    """
    function for loading from .json-file
    Args:
        filepath: filepath to -json-file

    Returns:
        json_data: dictionary containing the data from file
    """
    with open(filepath, "r") as f:
        json_data = json.load(f)
    return json_data


if __name__ == '__main__':
    reward = [1, -1, 0.3, 0]  # [win, lose, progress, no progress]
    filepath = "Result_file.json"  # Change to specify filepath for creation of .json result file
    num_episodes = 200
    batch_size = 64
    gamma = 0
    learning_rate = 1e-4
    HEIGHT = 3
    WIDTH = 3
    N_BOMBS = 1
    eps = 1.
    eps_end = 0.1
    eps_decay = eps/(num_episodes-num_episodes*0.1)
    conv = False  # Bool to specify if a CNN is to be used. If False a FC NN is instead used
    dim = HEIGHT

    # Create the environment
    env = MineSweeperEnv(HEIGHT, WIDTH, N_BOMBS, reward)

    # Initializations
    num_actions = env.n_actions
    num_states = env.observation_space.shape[0]

    # Choosing device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    R_buffer, R_avg, eps, avg_wins, i, ep_reward, R_avg_progress, wins = train_loop_ddqn(ddqn, env, replay_buffer,
                                                                                         num_episodes,
                                                                                         batch_size=batch_size,
                                                                                         gamma=gamma,
                                                                                         eps=eps,
                                                                                         eps_end=eps_end,
                                                                                         eps_decay=eps_decay)

    eps_list.extend(eps)
    avg_wins_list.extend(avg_wins)
    R_avg_list.extend(R_avg)
    wins_list.extend(wins)

    while True:
        inp = input("Continue?: ")
        inp = int(inp)
        if inp == 1:
            eps = input("Epsilon?: ")
            eps = float(eps)
            eps_end = input("Epsilon end?: ")
            eps_end = float(eps_end)
            epi = input("Number of episodes?: ")
            num_episodes = int(epi)
            eps_decay = eps / (num_episodes - num_episodes * 0.1)
            R_buffer, R_avg, eps, avg_wins, i, ep_reward, R_avg_progress, wins = train_loop_ddqn(ddqn, env,
                                                                                                 replay_buffer,
                                                                                                 num_episodes,
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
    moving_avg = compute_moving_average(result_dict["wins"])
    result_dict["moving_avg_wins"] = moving_avg

    write_to_json(result_dict, filepath)
    data = load_from_json(filepath)
    plot_reward(data)
