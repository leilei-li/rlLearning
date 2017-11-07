import numpy as np


def init_gridworld_matrix():
    '''
        init a 5*5 gridworld matrix.
        all of the selections of moving position,
        all of the reward of the selections;
    :return:
        gridworld_matrix: a 5*5 matrix fulled of zero,
        action_select_prod: record the probability of each elelctions,
        next_state: record the position after selecting a action,
        action_reward: record the reward of each action.

    '''
    positions = ['north', 'south', 'west', 'east']
    gridworld_matrix = np.zeros((5, 5))
    action_select_prod = []  # record every choice of each element in gridworld matrix
    for i in range(5):
        action_select_prod.append([])
        for j in range(5):
            action_select_prod[i].append({'north': 0.25, 'south': 0.25, 'west': 0.25, 'east': 0.25})
    next_state = []  # record next position of each element
    action_reward = []  # record every reward of each element's action
    for i in range(5):
        next_state.append([])
        action_reward.append([])
        for j in range(5):
            ele_next_position = {}
            ele_action_reward = {}
            # handle the boundary of the gridworld matrix
            if i == 0:
                ele_next_position['north'] = [i, j]
                ele_action_reward['north'] = -1.0
            else:
                ele_next_position['north'] = [i - 1, j]
                ele_action_reward['north'] = 0.0
            if i == 4:
                ele_next_position['south'] = [i, j]
                ele_action_reward['south'] = -1.0
            else:
                ele_next_position['south'] = [i + 1, j]
                ele_action_reward['south'] = 0.0
            if j == 0:
                ele_next_position['west'] = [i, j]
                ele_action_reward['west'] = -1.0
            else:
                ele_next_position['west'] = [i, j - 1]
                ele_action_reward['west'] = 0.0
            if j == 4:
                ele_next_position['east'] = [i, j]
                ele_action_reward['east'] = -1.0
            else:
                ele_next_position['east'] = [i, j + 1]
                ele_action_reward['east'] = 0.0

            # handle the special point A and B
            if i == 0 and j == 1:  # point A
                for position in positions:
                    ele_next_position[position] = [4, 1]
                    ele_action_reward[position] = 10.0
            if i == 0 and j == 3:  # point B
                for position in positions:
                    ele_next_position[position] = [2, 3]
                    ele_action_reward[position] = 5.0
            next_state[i].append(ele_next_position)
            action_reward[i].append(ele_action_reward)
    return gridworld_matrix, action_select_prod, next_state, action_reward


def iterative_value_matrix(gridworld_matrix, feed_list):
    action_select_prod = feed_list[0]
    next_state = feed_list[1]
    action_reward = feed_list[2]
    actions = ['north', 'south', 'west', 'east']
    next_gridworld_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            for action in actions:
                next_position = next_state[i][j][action]
                # accroding the random policy and its Bellman equation
                next_point_x = next_position[0]
                next_point_y = next_position[1]
                next_gridworld_matrix[i][j] = next_gridworld_matrix[i][j] + action_select_prod[i][j][action] * (
                    action_reward[i][j][action] + 0.9 * gridworld_matrix[next_point_x, next_point_y])
    return next_gridworld_matrix


def main():
    gridworld_matrix, action_select_prod, next_state, action_reward = init_gridworld_matrix()
    feed_list = [action_select_prod, next_state, action_reward]
    rounds = 0
    while True:
        rounds = rounds + 1
        next_gridworld_matrix = iterative_value_matrix(gridworld_matrix, feed_list)
        if np.sum(np.abs(next_gridworld_matrix - gridworld_matrix)) < 1e-6:
            print('after running: ', rounds)
            print(next_gridworld_matrix.round(1))
            break
        gridworld_matrix = next_gridworld_matrix


if __name__ == '__main__':
    main()
