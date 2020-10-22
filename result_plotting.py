from plot_functions import plot_reward
from main import load_from_json


def compute_moving_average(data):
    window_size = 100
    i = 0
    moving_averages = []
    while i < len(data) - window_size + 1:
        this_window = data[i: i + window_size]

        window_average = sum(this_window) / window_size*100
        moving_averages.append(window_average)
        i += 1
    return moving_averages


filepath = "Result_RANDOM_BOMBS_.json"
data = load_from_json(filepath)
if 'moving_avg_wins' not in data.keys():
    moving_avg = compute_moving_average(data["wins"])
    data["moving_avg_wins"] = moving_avg
plot_reward(data)

