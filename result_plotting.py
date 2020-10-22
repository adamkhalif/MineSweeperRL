from plot_functions import plot_reward
from main import load_from_json


filepath = "Result_RANDOM_BOMBS_poster.json"
data = load_from_json(filepath)
plot_reward(data)
