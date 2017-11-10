'''
Reference: https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/blob/master/chapter05/Blackjack.py
'''
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
figureIndex = 0

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


def target_policy_player(player_use_Ace_as_11, player_sum, dealerCard):
    '''
        get target policy of player
    :param player_sum: the sum of cards in player's hand
    :return: the player_policy when he has player_sum cards
    '''
    return player_policy[player_sum]


def behavior_player(player_use_Ace_as_11, player_sum, dealerCard):
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


def play(player_policy_FN, initial_state=None, initial_actions=None):
    '''
          simulate a Blackjack game.
    :param player_policy_FN: the specify policy that player choose.
    :param initial_state: a list :
                            [whether players has use Ace as 11,
                            the sum of player's cards,
                             one card of dealer]
    :param initial_actions: the initial action
    :return: state: just like initial_state;
    :return: reward: win:+1;    lose:-1;    draw:0;
    :return: palyer_trajectory: Monte Carlo trajectory of this Blackjack game.
    '''
    player_sum = 0
    player_trajectory = []
    player_use_Ace_as_11 = False  # player_use Ace as 11 when TRUE; as 1 when FALSE;
    dealer_card_1 = 0
    dealer_card_2 = 0
    dealer_use_Ace_as_11 = False
    if initial_state is None:
        # no initial state means using a random initial state.
        number_of_Ace = 0

        # init player's cards
        while player_sum < 12:
            # always get a card if sum<12
            card = get_a_card()

            # if get a Ace, use it as 11 at first.
            if card == 1:
                number_of_Ace = number_of_Ace + 1
                card = 11
                player_use_Ace_as_11 = True
            player_sum = player_sum + card

        if player_sum > 21:
            # if player_sum>21,he must hold at least 1 Ace(maybe 2 Aces),
            #   and he use Ace as 1 rather than 11.
            player_sum = player_sum - 10
            if number_of_Ace == 1:  # he only has 1 Ace
                player_use_Ace_as_11 = False

        # init two cards of dealer, and show the first to player.
        dealer_card_1 = get_a_card()
        dealer_card_2 = get_a_card()
    else:
        # use the specified initial state
        player_use_Ace_as_11 = initial_state[0]
        player_sum = initial_state[1]
        dealer_card_1 = initial_state[2]
        dealer_card_2 = get_a_card()

    # init the state of this game
    state = [player_use_Ace_as_11, player_sum, dealer_card_1]

    # init dealer's sum
    dealer_sum = 0
    if dealer_card_1 == 1 and dealer_card_2 != 1:
        dealer_sum = dealer_sum + 11 + dealer_card_2
        dealer_use_Ace_as_11 = True
    elif dealer_card_1 != 1 and dealer_card_2 == 1:
        dealer_sum = dealer_sum + dealer_card_1 + 11
        dealer_use_Ace_as_11 = True
    elif dealer_card_1 == 1 and dealer_card_2 == 1:
        dealer_sum = dealer_sum + 1 + 11
        dealer_use_Ace_as_11 = True
    else:
        dealer_sum = dealer_card_1 + dealer_card_2
        dealer_use_Ace_as_11 = False

    # start game:
    # player's turn
    while True:
        if initial_actions is not None:
            action = initial_actions
            initial_actions = None
        else:
            action = player_policy_FN(player_use_Ace_as_11, player_sum, dealer_card_1)

        # tracking player's trajectory
        player_trajectory.append([action, (player_use_Ace_as_11, player_sum, dealer_card_1)])
        if action == ACTION_STAND:
            break
        else:
            player_sum = player_sum + get_a_card()
            # player hits

        # if player busts
        if player_sum > 21:
            # if player has a Ace,use it as 1
            if player_use_Ace_as_11 == True:
                player_sum = player_sum - 10
                player_use_Ace_as_11 = False
            else:
                # player loses
                return state, -1, player_trajectory

    # dealer's turn
    while True:
        # get action based on current state in dealer_policy[sum]
        action = dealer_policy[dealer_sum]
        if action == ACTION_STAND:
            break
        else:
            dealer_sum = dealer_sum + get_a_card()
            # dealer hits, get a new card

        # if dealer busts
        if dealer_sum > 21:
            # just like player, use Ace as 1 rather than 11.
            if dealer_use_Ace_as_11 == True:
                dealer_sum = dealer_sum - 10
                dealer_use_Ace_as_11 = False
            else:
                # dealer lose this game
                return state, 1, player_trajectory

    # both dealer and player strike,and both of them less than 21.
    # compare player_sum and dealer_sum,the bigger is winner.
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    if player_sum < dealer_sum:
        return state, -1, player_trajectory
    if player_sum == dealer_sum:
        return state, 0, player_trajectory


def monte_carlo_on_policy(n_eposodes):
    '''
        Monte Carlo Method with on policy;
    :param n_eposodes:
    :return:
    '''
    statesUsableAce = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    statesUsableAceCount = np.ones((10, 10))
    statesNoUsableAce = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    statesNoUsableAceCount = np.ones((10, 10))
    for i in range(n_eposodes):
        state, reward, trajectory = play(target_policy_player)
        state[1] = state[1] - 12
        state[2] = state[2] - 1
        if state[0]:
            # player use Ace as 11
            statesUsableAceCount[state[1], state[2]] += 1
            statesUsableAce[state[1], state[2]] += reward
        else:
            statesNoUsableAceCount[state[1], state[2]] += 1
            statesNoUsableAce[state[1], state[2]] += reward
    return statesUsableAce / statesUsableAceCount, statesNoUsableAce / statesNoUsableAceCount


def prettyPrint(data, tile, zlabel='reward'):
    '''
        draw a 3D plot figure
    :param data:
    :param tile:
    :param zlabel:
    :return:
    '''
    global figureIndex
    fig = plt.figure(figureIndex)
    figureIndex += 1
    fig.suptitle(tile)
    ax = fig.add_subplot(111, projection='3d')
    axisX = []
    axisY = []
    axisZ = []
    # DX = []
    # DY = []
    for i in range(12, 22):
        for j in range(1, 11):
            axisX.append(i)
            axisY.append(j)
            axisZ.append(data[i - 12, j - 1])
            # if data[i - 12, j - 1] == 0:
            #     DX.append(i)
            #     DY.append(j)
    ax.scatter(axisY, axisX, axisZ)
    ax.set_ylabel('player sum')
    ax.set_xlabel('dealer showing')
    ax.set_zlabel(zlabel)
    # plt.figure()
    # plt.xlabel('dealer showing')
    # plt.ylabel('player sum')
    # for i in range(len(DX)):
    #     plt.scatter(DY[i],DX[i])
    # plt.show()


def onPolicy():
    '''
        run a monte carlo method
    :return:
    '''
    statesUsableAce1, statesNoUsableAce1 = monte_carlo_on_policy(10000)
    statesUsableAce2, statesNoUsableAce2 = monte_carlo_on_policy(500000)
    # iterative Blackjack with 10000 rounds or 500000 rounds.
    prettyPrint(statesUsableAce1, 'Usable Ace, After 10000 Episodes')
    prettyPrint(statesNoUsableAce1, 'No Usable Ace, After 10000 Episodes')
    prettyPrint(statesUsableAce2, 'Usable Ace, After 500000 Episodes')
    prettyPrint(statesNoUsableAce2, 'No Usable Ace, After 500000 Episodes')
    plt.show()


def monteCarloOffPolicy(nEpisodes):
    initialState = [True, 13, 2]
    sumOfImportanceRatio = [0]
    sumOfRewards = [0]
    for i in range(0, nEpisodes):
        state, reward, playerTrajectory = play(behavior_player, initial_state=initialState)

        # get the importance ratio
        importanceRatioAbove = 1.0
        importanceRatioBelow = 1.0
        for action, (usableAce, playerSum, dealerCard) in playerTrajectory:
            if action == target_policy_player(usableAce, playerSum, dealerCard):
                importanceRatioBelow *= 0.5
            else:
                importanceRatioAbove = 0.0
                break
        # just like on-policy iterative this game
        importanceRatio = importanceRatioAbove / importanceRatioBelow
        sumOfImportanceRatio.append(sumOfImportanceRatio[-1] + importanceRatio)
        sumOfRewards.append(sumOfRewards[-1] + reward * importanceRatio)
    del sumOfImportanceRatio[0]
    del sumOfRewards[0]

    sumOfRewards = np.asarray(sumOfRewards)
    sumOfImportanceRatio = np.asarray(sumOfImportanceRatio)
    ordinarySampling = sumOfRewards / np.arange(1, nEpisodes + 1)

    with np.errstate(divide='ignore', invalid='ignore'):  # avoid dividing 0
        weightedSampling = np.where(sumOfImportanceRatio != 0, sumOfRewards / sumOfImportanceRatio, 0)

    return ordinarySampling, weightedSampling


def offPolicy():
    trueValue = -0.27726
    nEpisodes = 10000
    nRuns = 100
    ordinarySampling = np.zeros(nEpisodes)
    weightedSampling = np.zeros(nEpisodes)
    for i in range(0, nRuns):
        ordinarySampling_, weightedSampling_ = monteCarloOffPolicy(nEpisodes)
        # get the squared error
        ordinarySampling += np.power(ordinarySampling_ - trueValue, 2)
        weightedSampling += np.power(weightedSampling_ - trueValue, 2)
    ordinarySampling /= nRuns
    weightedSampling /= nRuns
    # draw fig 5.4
    axisX = np.log10(np.arange(1, nEpisodes + 1))
    plt.plot(axisX, ordinarySampling, label='Ordinary Importance Sampling')
    plt.plot(axisX, weightedSampling, label='Weighted Importance Sampling')
    plt.xlabel('Episodes')
    plt.ylabel('Mean square error')
    plt.ylim(0, 4)
    plt.legend()
    plt.show()


def monteCarloES(nEpisodes):
    # (playerSum, dealerCard, usableAce, action)
    stateActionValues = np.zeros((10, 10, 2, 2))
    # initialze counts to 1 to avoid division by 0
    stateActionPairCount = np.ones((10, 10, 2, 2))

    # behavior policy is greedy
    def behaviorPolicy(usableAce, playerSum, dealerCard):
        usableAce = int(usableAce)
        playerSum -= 12
        dealerCard -= 1
        # get argmax of the average returns(s, a)
        values_ = stateActionValues[playerSum, dealerCard, usableAce, :] / stateActionPairCount[playerSum, dealerCard,
                                                                           usableAce, :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # play for several episodes
    for episode in range(nEpisodes):
        if episode % 1000 == 0:
            print('episode:', episode)
        # for each episode, use a randomly initialized state and action
        initialState = [bool(np.random.choice([0, 1])),
                        np.random.choice(range(12, 22)),
                        np.random.choice(range(1, 11))]
        initialAction = np.random.choice(actions)
        _, reward, trajectory = play(behaviorPolicy, initialState, initialAction)
        for action, (usableAce, playerSum, dealerCard) in trajectory:
            usableAce = int(usableAce)
            playerSum -= 12
            dealerCard -= 1
            # update values of state-action pairs
            stateActionValues[playerSum, dealerCard, usableAce, action] += reward
            stateActionPairCount[playerSum, dealerCard, usableAce, action] += 1

    return stateActionValues / stateActionPairCount


def monte_carlo_exploring_starts():
    stateActionValues = monteCarloES(500000)
    stateValueUsableAce = np.zeros((10, 10))
    stateValueNoUsableAce = np.zeros((10, 10))
    # get the optimal policy
    actionUsableAce = np.zeros((10, 10), dtype='int')
    actionNoUsableAce = np.zeros((10, 10), dtype='int')
    for i in range(10):
        for j in range(10):
            stateValueNoUsableAce[i, j] = np.max(stateActionValues[i, j, 0, :])
            stateValueUsableAce[i, j] = np.max(stateActionValues[i, j, 1, :])
            actionNoUsableAce[i, j] = np.argmax(stateActionValues[i, j, 0, :])
            actionUsableAce[i, j] = np.argmax(stateActionValues[i, j, 1, :])
    prettyPrint(stateValueUsableAce, 'Optimal state value with usable Ace')
    prettyPrint(stateValueNoUsableAce, 'Optimal state value with no usable Ace')
    prettyPrint(actionUsableAce, 'Optimal policy with usable Ace', 'Action (0 Hit, 1 Stick)')
    prettyPrint(actionNoUsableAce, 'Optimal policy with no usable Ace', 'Action (0 Hit, 1 Stick)')
    plt.show()


# onPolicy()
# monte_carlo_exploring_starts()
offPolicy()
