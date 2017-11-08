import numpy as np
import matplotlib.pyplot as plt


def init_feed_dict():
    '''
        init some global parameters.
    :return:
        a feed_dict which not changed in whole algorithm.
    '''
    feed_dict = {}
    goal = 100  # how many coins that gamblers hold
    feed_dict['goal'] = goal
    states = np.arange(0, goal + 1)  # state 0 to state 100
    feed_dict['states'] = states
    head_prob = 0.4
    feed_dict['head_prob'] = head_prob
    policy = np.zeros(goal + 1)  # init all policy to 0 which mapping states
    feed_dict['policy'] = policy
    state_value = np.zeros(goal + 1)  # init state_value to 0
    state_value[goal] = 1  # if gambles got the goal,the reward is +1
    feed_dict['state_value'] = state_value
    return feed_dict


def iteration_state_value_AND_optimal_policy(feed_dict):
    states = feed_dict['states']
    goal = feed_dict['goal']
    head_prob = feed_dict['head_prob']
    state_value = feed_dict['state_value']
    policy = feed_dict['policy']
    # iteration state value
    while True:
        delta = 0
        for state in states[1:goal]:
            actions = np.arange(min(state, goal - state) + 1)  # choose how many coins heads
            value_list = []
            for action in actions:
                current_value = head_prob * state_value[state + action] + (1 - head_prob) * state_value[state - action]
                value_list.append(current_value)
            new_value = np.max(value_list)
            delta = delta + np.abs(state_value[state] - new_value)
            state_value[state] = new_value
        if delta < 1e-6:
            break
    # caculate optimal_policy with the optimal state value matrix
    for state in states[1:goal]:
        actions = np.arange(min(state, goal - state) + 1)
        value_list = []
        for action in actions:
            current_value = head_prob * state_value[state + action] + (1 - head_prob) * state_value[state - action]
            value_list.append(current_value)
        policy[state] = actions[np.argmax(value_list)]

    return state_value, states, policy


def draw_fig(state_value, states, policy):
    plt.figure()
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax1.grid(True)
    ax1.set_title('state value')
    ax1.set_xlabel('capital')
    ax1.set_ylabel('value estimates')
    ax1.plot(state_value)
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    ax2.grid(True)
    ax2.set_title('optimal policy')
    ax2.set_xlabel('capital')
    ax2.set_ylabel('final policy')
    ax2.scatter(states, policy)
    plt.show()

def main():
    feed_dict = init_feed_dict()
    state_value, states, policy = iteration_state_value_AND_optimal_policy(feed_dict)
    draw_fig(state_value, states, policy)


if __name__ == '__main__':
    main()
