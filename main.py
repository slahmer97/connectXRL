from kaggle_environments import evaluate, make, utils
import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

"""
NN specification : 
NN used as value-action function approximation.

Input : array of size (6*7) that represents the game state (grid)
Output : array of size (7) which represents a value for each of 7 possible actions
NN function can be formulated mathematically by : 
    NN(state) = output 
              = [value_action_0;value_action_1;value_action_2;....;value_action_6]
    where : 
        @state is the input given to the NN
        @output represents : [Q^(state,action_0);Q^(state,action_1);....;Q^(state,action_6)]
        where :
            Q^(state,action) is the estimated value-action function for a give pair of state and action
Upon getting a new observation, the agent will feed the current state as input to NN, then apply action
that has the maximum value.
"""

env = make("connectx", debug=True)


def my_agent(observation, configuration):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])


def get_possible_actions(board):
    tmp = np.reshape(board, (6, 7))
    for i in tmp:
        for j in i:
            print("{} ".format(j), end="")
        print()
    res = []
    return res


# Model builder
def model_builder(input_size=7 * 6, action_size=7, lrt=0.008):
    # TODO improve the model
    model = Sequential()
    model.add(Dense(24, input_dim=input_size, activation='relu'))
    model.add(Dense(24, activation='relu'))

    # TODO make sure to use the right activation function for output layer
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=lrt))  # TODO tune lr, check other loss functions
    return model


data_set = deque(maxlen=5000)

env.reset()
trainer = env.train([None, "random"])
t = np.reshape(trainer.reset().board, [1, 7 * 6])

model = model_builder()
predicted = model.predict(t)
action = 0
observation, reward, done, info = trainer.step(action)
predicted[0][action] = -100 + 0.8 * np.amax(model.predict(np.reshape(observation.board, [1, 7 * 6]))[0])
print(predicted)
model.fit(t, predicted, epochs=1)
print(model.predict(t)[0])

exit(1)

trainer.reset()
observation, reward, done, info = trainer.step(0)
get_possible_actions(observation.board)
exit()
while not env.done:
    my_action = my_agent(observation, env.configuration)

    next_state, reward, done, info = trainer.step(my_action)
    data_set.
    print("-------------- My Action {} : ".format(my_action), info, done, reward)

    print(env.render(mode="ansi"))
