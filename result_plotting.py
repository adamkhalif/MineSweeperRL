from plot_functions import plot_reward
from main import load_from_json


"""
This file plots the results from a .json-file. 
change filepath to desired .json-file

Only run this file if you would like to plot an already existing .json-file
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

# Try setting filepath equal to one of these to test this script
# "Result_random_agent.json"
# "Result_FIXED_BOMBS.json"
# "Result_RANDOM_BOMBS_no_lose_first_15k.json"
filepath = "Result_random_agent.json"
data = load_from_json(filepath)
if 'moving_avg_wins' not in data.keys():
    moving_avg = compute_moving_average(data["wins"])
    data["moving_avg_wins"] = moving_avg
plot_reward(data)

