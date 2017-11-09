import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# some golbal parameter

'''
def two actions in the game.
    ACTION_HIT: means the player ask another cards;
    ACTION_STAND: means he stops get cards(sticks).
'''
ACTION_HIT = 0
ACTION_STAND = 1  # strike
actions = [ACTION_HIT, ACTION_STAND]

'''
init players policy,def his value function is v(s): 
    if v(s) in [12,20),he will hit;
    if v(s) == 20 or v(s)== 21,he will strike;
    if v(s) < 12, he will hit a card automatically.
'''
player_policy = np.zeros(22)
for i in range(12, 20):
    player_policy[i] = ACTION_HIT
player_policy[20] = ACTION_STAND
player_policy[21] = ACTION_STAND

'''
init dealer policy,def his value function is v(s): 
    if v(s) in [12,17),he will hit;
    if v(s) in [17,22),he will strike;
'''
dealer_policy = np.zeros(22)
for i in range(12, 17):
    dealer_policy[i] = ACTION_HIT
for i in range(17, 22):
    dealer_policy[i] = ACTION_STAND


def target_policy_player(player_sum):
    '''
        get target policy of player
    :param player_sum: the sum of cards in player's hand
    :return: the player_policy when he has player_sum cards
    '''
    return player_policy[player_sum]


def behavior_player():
    '''
        player has 50% probability to hit and 50% to stand
    :return: ACTION_HIT or ACTION_STAND
    '''
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STAND
    return ACTION_HIT


def get_a_card():
    '''
        randomly get a new card
    :return: a number in [1,10]
    '''
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card
