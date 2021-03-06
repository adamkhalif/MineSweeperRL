import numpy as np
from collections import namedtuple
import torch
import random

"""
This file contains all necessary functions to be able to make use of Q-learning. 
The functions have been copied from the pre defined functions in HA3 and also our own creations in HA3.
"""


def eps_greedy_policy(q_values, eps, forbidden_actions):
    """
    Creates an epsilon-greedy policy
    :param q_values: set of Q-values of shape (num actions,)
    :param eps: probability of taking a uniform random action
    :param forbidden_actions: list with Bools containing actions already chosen.
    :return: policy of shape (num actions,)
    """

    q_values[forbidden_actions] = np.NINF
    indices = torch.nonzero(q_values == q_values.max())
    random_index = random.randint(0, indices.shape[1]-1)
    best_action_index = indices[random_index]
    l = len(q_values)
    n_forbidden_actions = np.count_nonzero(forbidden_actions)
    p = eps / (l-n_forbidden_actions)

    policy = np.full([l], p)
    policy[forbidden_actions] = 0
    policy[best_action_index] += 1 - eps

    return policy


def calc_q_and_take_action(ddqn, state, eps, forbidden_actions):
    '''
    Calculate Q-values for current state, and take an action according to an epsilon-greedy policy.
    Inputs:
        ddqn   - DDQN model. An object holding the online / offline Q-networks, and some related methods.
        state  - Current state. Numpy array, shape (1, num_states).
        eps    - Exploration parameter.
    Returns:
        q_online_curr   - Q(s,a) for current state s. Numpy array, shape (1, num_actions) or  (num_actions,).
        curr_action     - Selected action (0 or 1, i.e., left or right), sampled from epsilon-greedy policy. Integer.
    '''
    # FYI:
    # ddqn.online_model & ddqn.offline_model are Pytorch modules for online / offline Q-networks,
    # which take the state as input, and output the Q-values for all actions.
    # Input shape (batch_size, num_states). Output shape (batch_size, num_actions).

    state = torch.Tensor(state).detach().flatten()
    state = state.to(device=ddqn._device)
    q_online_curr = ddqn.online_model(state)

    pi = eps_greedy_policy(q_online_curr, eps, forbidden_actions)

    curr_action = np.random.choice(range(len(pi)), p=pi)

    return q_online_curr, curr_action


def calculate_q_targets(q1_batch, q2_batch, r_batch, nonterminal_batch, gamma=.99):
    """
    Calculates the Q target used for the loss
    : param q1_batch: Batch of Q(s', a) from online network. FloatTensor, shape (N, num actions)
    : param q2_batch: Batch of Q(s', a) from target network. FloatTensor, shape (N, num actions)
    : param r_batch: Batch of rewards. FloatTensor, shape (N,)
    : param nonterminal_batch: Batch of booleans, with False elements if state s' is terminal and True otherwise. BoolTensor, shape (N,)
    : param gamma: Discount factor, float.
    : return: Q target. FloatTensor, shape (N,)
    """

    actions = torch.argmax(q1_batch, dim=1)

    max_q2 = q2_batch[torch.arange(q2_batch.size(0)), actions]

    Y = r_batch + gamma * max_q2

    Y[~nonterminal_batch] = r_batch[~nonterminal_batch]

    return Y


def sample_batch_and_calculate_loss(ddqn, replay_buffer, batch_size, gamma):
    '''
    Sample mini-batch from replay buffer, and compute the mini-batch loss
    Inputs:
        ddqn          - DDQN model. An object holding the online / offline Q-networks, and some related methods.
        replay_buffer - Replay buffer object (from which samples will be drawn)
        batch_size    - Batch size
        gamma         - Discount factor
    Returns:
        Mini-batch loss, on which .backward() will be called to compute gradient.
    '''
    # Sample a minibatch of transitions from replay buffer
    curr_state, curr_action, reward, next_state, nonterminal = replay_buffer.sample_minibatch(batch_size)

    # FYI:
    # ddqn.online_model & ddqn.offline_model are Pytorch modules for online / offline Q-networks,
    # which take the state as input, and output the Q-values for all actions.
    # Input shape (batch_size, num_states). Output shape (batch_size, num_actions).

    q_online_curr = ddqn.online_model(curr_state)
    q_online_next = ddqn.online_model(next_state)
    with torch.no_grad():
        q_offline_next = ddqn.offline_model(next_state)

    q_target = calculate_q_targets(q_online_next, q_offline_next, reward, nonterminal, gamma=gamma)
    loss = ddqn.calc_loss(q_online_curr, q_target, curr_action)

    return loss


def train_loop_ddqn(ddqn, env, replay_buffer, num_episodes, batch_size=64, gamma=.94, eps=1., eps_end=0, eps_decay=.001):
    Transition = namedtuple("Transition", ["s", "a", "r", "next_s", "t"])

    tau = 1000
    cnt_updates = 0
    R_buffer = []
    R_avg = []
    wins = []
    R_eps = []
    R_avg_wins = []
    R_i = []
    R_ep_reward = []
    R_clickable_boxes = []
    R_avg_progress = []
    for i in range(num_episodes):
        state = env.reset() # Initial state
        state = state[None, :]  # Add singleton dimension, to represent as batch of size 1.
        finish_episode = False  # Initialize
        ep_reward = 0  # Initialize "Episodic reward", i.e. the total reward for episode, when disregarding discount factor.
        q_buffer = []
        steps = 0

        while not finish_episode:
            steps += 1

            # Take one step in environment. No need to compute gradients,
            # we will just store transition to replay buffer, and later sample a whole batch
            # from the replay buffer to actually take a gradient step.
            forbidden_actions = env.forbidden_actions
            q_online_curr, curr_action = calc_q_and_take_action(ddqn, state, eps, forbidden_actions)
            q_buffer.append(q_online_curr)
            if env.first_move:
                curr_action = 8
                env.first_move = False
            new_state, reward, finish_episode, _ = env.step(curr_action)  # take one step in the environment
            new_state = new_state[None, :]

            # Assess whether terminal state was reached.
            # The episode may end due to having reached 200 steps, but we should not regard this as reaching the
            # terminal state, and hence not disregard Q(s',a) from the Q target.
            # https://arxiv.org/abs/1712.00378
            nonterminal_to_buffer = not finish_episode or steps == 200

            # Store experienced transition to replay buffer
            replay_buffer.add(Transition(s=state, a=curr_action, r=reward, next_s=new_state, t=nonterminal_to_buffer))

            state = new_state
            ep_reward += reward

            # If replay buffer contains more than 1000 samples, perform one training step
            if replay_buffer.buffer_length > 1000:
                loss = sample_batch_and_calculate_loss(ddqn, replay_buffer, batch_size, gamma)
                ddqn.optimizer.zero_grad()
                loss.backward()
                ddqn.optimizer.step()

                cnt_updates += 1
                if cnt_updates % tau == 0:
                    ddqn.update_target_network()

        eps = max(eps - eps_decay, eps_end)  # decrease epsilon
        R_buffer.append(ep_reward)

        # Running average of episodic rewards (total reward, disregarding discount factor)
        R_avg.append(.05 * R_buffer[i] + .95 * R_avg[i - 1]) if i > 0 else R_avg.append(R_buffer[i])

        if env.WIN:
            wins.append(1)
        else:
            wins.append(0)
        R_eps.append(eps)
        R_i.append(i)
        R_ep_reward.append(ep_reward)
        R_clickable_boxes.append(env.n_not_bombs_left)
        if i < 100:
            R_avg_wins.append(0)
        else:
            avg_wins = sum(wins[-100::])
            R_avg_wins.append(avg_wins)
            avg_progress = sum(R_clickable_boxes[-100::])/100
            R_avg_progress.append(avg_progress)
            print('Epsilon: {:.3f}, Win rate: {:.2f}%, Episode {:d}, Clickable boxes left: {:d}, Win?: {:1d},'
                  ' ''Reward: {:.1f}'.format(eps, avg_wins, i, env.n_not_bombs_left, env.WIN, ep_reward))

    return R_buffer, R_avg, R_eps, R_avg_wins, R_i, R_ep_reward, R_avg_progress, wins
