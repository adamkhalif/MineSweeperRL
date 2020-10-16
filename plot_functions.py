
import matplotlib.pyplot as plt

def plot_reward(R, R_avg):
    rewards = plt.plot(R, alpha=.4, label='R')
    avg_rewards = plt.plot(R_avg,label='avg R')
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Episode')
    plt.ylim(-15, 8)
    plt.show()