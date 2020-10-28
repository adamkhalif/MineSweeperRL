import gym
from gym import spaces
import numpy as np
import random


class MineSweeperEnv(gym.Env):
    """
    Class that creates the game environment. Skeleton taken from stable-baselines:
    https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html

    """
    def __init__(self, HEIGHT=9, WIDTH=9, N_BOMBS=10,reward=[100, -300, 50, -0]):
        super().__init__()
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.N_BOMBS = N_BOMBS
        self.action_space = spaces.Discrete(WIDTH * HEIGHT)
        self.n_actions = HEIGHT * WIDTH
        self.state = np.array(HEIGHT * WIDTH)
        self.bomb_env = np.array((HEIGHT, WIDTH))
        self.observation_space = spaces.Box(np.full(HEIGHT * WIDTH, -2), np.full(HEIGHT * WIDTH, min(8,self.N_BOMBS)), dtype=np.int)
        self.n_not_bombs_left = 0
        self.reward = reward  # [win, lose, progress, no progress]

        self.RANDOM_BOMBS = True  # Set to false if you want to evaluate a fixed bomb placement
        self.n_wins = 0
        self.WIN = False
        self.first_move = True
        self.forbidden_actions = np.full((WIDTH * HEIGHT), False)

    def step(self, action):
        """
        Calculates the reward for a given action. Also
        Args:
            action: Current action chosen by agent

        Returns:
            observation: new state space
            reward: reward for chosen action
            done: bool if game is over or not
        """
        reward = 0
        self.forbidden_actions[action] = True
        self.WIN = False
        row = action // self.HEIGHT
        col = action % self.WIDTH
        done = False
        if self.bomb_env[row, col] == -2:
            reward = self.reward[1]  # loose
            done = True
        elif self.state[action] < 0:
            if not done:
                reward = self.reward[2]  # progress
                self.n_not_bombs_left -= 1
                if self.n_not_bombs_left == 0:
                    done = True
                    reward = self.reward[0]  # win
                    self.n_wins += 1
                    self.WIN = True
        else:
            reward = self.reward[3]  # no progress
            done = True

        self.state[action] = self.bomb_env[row, col]

        observation = self.state
        return observation, reward, done, {}

    def reset(self):
        """
        Resets the game and forces the bomb to never be placed in the cell that corresponds with the first action
        Returns:
            self.state: reset state space

        """
        self.first_move = True
        self.forbidden_actions = np.full((self.WIDTH * self.HEIGHT), False)
        self.n_not_bombs_left = self.HEIGHT * self.WIDTH - self.N_BOMBS
        self.bomb_env = np.zeros((self.HEIGHT, self.WIDTH))
        self.state = np.full(self.HEIGHT * self.WIDTH, -1)
        # set bombs
        if self.RANDOM_BOMBS:
            for b in range(self.N_BOMBS):
                self.bomb_env[random.randint(0, self.HEIGHT - 1), random.randint(0, self.WIDTH - 1)] = -2
                if self.bomb_env[2, 2] == -2:
                    self.reset()  # This is done to prevent the agent from losing on the first action
        else:
            self.bomb_env[0, 0] = -2
        self.sumNeighbours()

        return self.state

    def checkNeighbours(self, row, col):
        """
        Checks to see if there are any bombs around cell given by row, col
        Args:
            row: row value of action
            col: column value of action

        Returns:
            bomb_counter: The number of bombs around the cell
        """
        bomb_counter = 0
        for r in range(row - 1, row + 2):
            for c in range(col - 1, col + 2):
                if not (r < 0 or c < 0 or r > self.HEIGHT - 1 or c > self.WIDTH - 1):
                    if self.bomb_env[r, c] == -2:
                        bomb_counter += 1
        return bomb_counter

    def sumNeighbours(self):
        """
            Computes the total number of bombs around each cell in the state space
        Returns:
            -
        """
        for r in range(0, self.HEIGHT):
            for c in range(0, self.WIDTH):
                if not self.bomb_env[r, c] == -2:
                    self.bomb_env[r, c] = self.checkNeighbours(r, c)
