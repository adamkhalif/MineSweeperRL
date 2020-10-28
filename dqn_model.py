import torch
from torch import nn
import numpy as np
from collections import deque

"""
Much code in this file is copied from HA3.
the class QNetworkConv(nn.Module) have been added to be able to evaluate performance when using a CNN instead of only FC
"""


class ExperienceReplay:
    def __init__(self, device, num_states, buffer_size=1e+6):
        self._device = device
        self.__buffer = deque(maxlen=int(buffer_size))
        self._num_states = num_states

    @property
    def buffer_length(self):
        return len(self.__buffer)

    def add(self, transition):
        """
        Adds a transition <s, a, r, s', t > to the replay buffer
        :param transition:
        :return:
        """
        self.__buffer.append(transition)

    def sample_minibatch(self, batch_size=128):
        """
        :param batch_size:
        :return:
        """
        ids = np.random.choice(a=self.buffer_length, size=batch_size)
        state_batch = np.zeros([batch_size, self._num_states],
                               dtype=np.float32)
        action_batch = np.zeros([
            batch_size,
        ], dtype=np.int64)
        reward_batch = np.zeros([
            batch_size,
        ], dtype=np.float32)
        nonterminal_batch = np.zeros([
            batch_size,
        ], dtype=np.bool)
        next_state_batch = np.zeros([batch_size, self._num_states],
                                    dtype=np.float32)
        for i, index in zip(range(batch_size), ids):
            state_batch[i, :] = self.__buffer[index].s
            action_batch[i] = self.__buffer[index].a
            reward_batch[i] = self.__buffer[index].r
            nonterminal_batch[i] = self.__buffer[index].t
            next_state_batch[i, :] = self.__buffer[index].next_s

        return (
            torch.tensor(state_batch, dtype=torch.float, device=self._device),
            torch.tensor(action_batch, dtype=torch.long, device=self._device),
            torch.tensor(reward_batch, dtype=torch.float, device=self._device),
            torch.tensor(next_state_batch,
                         dtype=torch.float,
                         device=self._device),
            torch.tensor(nonterminal_batch,
                         dtype=torch.bool,
                         device=self._device),
        )


class QNetworkConv(nn.Module):
    """
    Added class to be able to evaluate performance using a CNN instead of a FC.
    To utilize this class set param: 'conv = True' in main.py
    Main structure of this class is inspired from class 'QNetwork(nn.Module)'.
    """
    def __init__(self, num_states, num_actions, dim=3):
        super().__init__()
        self.dim = dim
        self._num_states = num_states
        self._num_actions = num_actions
        self._norm1 = nn.LayerNorm(self._num_states)
        self._conv = nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self._relu1 = nn.ReLU(inplace=True)
        self._fc1 = nn.Linear(self._num_states, 100)
        self._relu2 = nn.ReLU(inplace=True)
        self._fc_final = nn.Linear(100, self._num_actions)

        # Initialize all bias parameters to 0, according to old Keras implementation
        nn.init.zeros_(self._fc1.bias)
        nn.init.zeros_(self._conv.bias)
        nn.init.zeros_(self._fc_final.bias)
        # Initialize final layer uniformly in [-1e-6, 1e-6] range, according to old Keras implementation
        nn.init.uniform_(self._fc_final.weight, a=-1e-6, b=1e-6)

    def forward(self, state):
        h = self._norm1(state)
        h = state.reshape(-1,1, self.dim, self.dim)
        h = self._relu1(self._conv(h))
        h = h.view(-1, self.dim*self.dim)
        h = self._norm1(h)
        h = self._relu2(self._fc1(h))
        q_values = self._fc_final(h)
        if q_values.shape[0] == 1:
            q_values = q_values.view(-1)

        return q_values


class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super().__init__()
        self._num_states = num_states
        self._num_actions = num_actions

        self._norm1 = nn.LayerNorm(self._num_states)
        self._fc1 = nn.Linear(self._num_states, 100)
        self._relu1 = nn.ReLU(inplace=True)
        self._norm2 = nn.LayerNorm(100)
        self._fc2 = nn.Linear(100, 60)
        self._norm3 = nn.LayerNorm(60)
        self._relu2 = nn.ReLU(inplace=True)
        self._fc_final = nn.Linear(60, self._num_actions)

        # Initialize all bias parameters to 0, according to old Keras implementation
        nn.init.zeros_(self._fc1.bias)
        nn.init.zeros_(self._fc2.bias)
        nn.init.zeros_(self._fc_final.bias)
        # Initialize final layer uniformly in [-1e-6, 1e-6] range, according to old Keras implementation
        nn.init.uniform_(self._fc_final.weight, a=-1e-6, b=1e-6)

    def forward(self, state):
        h = self._norm1(state)
        h = self._relu1(self._fc1(h))
        h = self._norm2(h)
        h = self._relu2(self._fc2(h))
        h = self._norm3(h)
        q_values = self._fc_final(h)

        return q_values


class DoubleQLearningModel(object):
    def __init__(self, device, num_states, num_actions, learning_rate, conv=False, dim=3):
        self._device = device
        self._num_states = num_states
        self._num_actions = num_actions
        self._lr = learning_rate

        # Define the two deep Q-networks
        if conv:
            self.online_model = QNetworkConv(self._num_states, self._num_actions, dim=dim).to(device=self._device)
            self.offline_model = QNetworkConv(self._num_states, self._num_actions, dim=dim).to(device=self._device)

        else:
            self.online_model = QNetwork(self._num_states, self._num_actions).to(device=self._device)
            self.offline_model = QNetwork(self._num_states, self._num_actions).to(device=self._device)

        # Define optimizer. Should update online network parameters only.
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self._lr, eps=1e-4)

        # Define loss function
        self._mse = nn.MSELoss(reduction='mean').to(device=self._device)

    def calc_loss(self, q_online_curr, q_target, a):
        """
        Calculate loss for given batch
        :param q_online_curr: batch of q values at current state. Shape (N, num actions)
        :param q_target: batch of temporal difference targets. Shape (N,)
        :param a: batch of actions taken at current state. Shape (N,)
        :return: calculated loss
        """
        batch_size = q_online_curr.shape[0]
        assert q_online_curr.shape == (batch_size, self._num_actions)
        assert q_target.shape == (batch_size,)
        assert a.shape == (batch_size,)

        # Select only the Q-values corresponding to the actions taken (loss should only be applied for these)
        q_online_curr_allactions = q_online_curr
        q_online_curr = q_online_curr[torch.arange(batch_size),a]  # New shape: (batch_size,)
        assert q_online_curr.shape == (batch_size,)
        for j in [0, 3, 4]:
            assert q_online_curr_allactions[j, a[j]] == q_online_curr[j]

        # Make sure that gradient is not back-propagated through Q target
        assert not q_target.requires_grad

        loss = self._mse(q_online_curr, q_target)
        assert loss.shape == ()

        return loss

    def update_target_network(self):
        """
        Update target network parameters, by copying from online network.
        """
        online_params = self.online_model.state_dict()
        self.offline_model.load_state_dict(online_params)


