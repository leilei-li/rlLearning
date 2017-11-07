import numpy as np

WORLD_SIZE = 4
REWARD = -1.0
ACTION_PROB = 0.25

world = np.zeros((WORLD_SIZE, WORLD_SIZE))

# left, up, right, down
actions = ['L', 'U', 'R', 'D']

nextState = []
for i in range(0, WORLD_SIZE):
    nextState.append([])
    for j in range(0, WORLD_SIZE):
        next = dict()
        if i == 0:
            next['U'] = [i, j]
        else:
            next['U'] = [i - 1, j]

        if i == WORLD_SIZE - 1:
            next['D'] = [i, j]
        else:
            next['D'] = [i + 1, j]

        if j == 0:
            next['L'] = [i, j]
        else:
            next['L'] = [i, j - 1]

        if j == WORLD_SIZE - 1:
            next['R'] = [i, j]
        else:
            next['R'] = [i, j + 1]

        nextState[i].append(next)

states = []
for i in range(0, WORLD_SIZE):
    for j in range(0, WORLD_SIZE):
        if (i == 0 and j == 0) or (i == WORLD_SIZE - 1 and j == WORLD_SIZE - 1):
            continue
        else:
            states.append([i, j])

# for figure 4.1
while True:
    # keep iteration until convergence
    newWorld = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for i, j in states:
        for action in actions:
            newPosition = nextState[i][j][action]
            # bellman equation
            newWorld[i, j] += ACTION_PROB * (REWARD + world[newPosition[0], newPosition[1]])
            if i==1 and j==0:
                a=newWorld[i][j]
                b=world[newPosition[0], newPosition[1]]
                x=newPosition[0]
                y=newPosition[1]
                print(a)
    if np.sum(np.abs(world - newWorld)) < 1e-6:
        print('Random Policy')
        print(newWorld.round(1))
        break
    world = newWorld