from plot_functions import plot_reward
from main import load_from_json
import json

filepath = "Result_RANDOM_BOMBS_no_lose_first.json"
data = load_from_json(filepath)
R_avg = 0
plot_reward(data)