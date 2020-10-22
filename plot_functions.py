
import matplotlib.pyplot as plt

def plot_reward(data):
    # First line
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Epsilon', color=color)
    ax1.plot(data["epsilon"], color=color, label="Epsilon")
    ax1.tick_params(axis='y', labelcolor=color)
    # Second line
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Winrate %', color=color)
    ax2.plot(data["moving_avg_wins"], color=color, label="Average winrate")
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    # Add legend
    lines = []
    labels = []

    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines, labels, loc="upper center")

    plt.show()

