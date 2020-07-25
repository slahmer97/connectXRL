import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class RLAgent:

    def __init__(self):
        self.tmp = 0
        self.data_set = deque(maxlen=5000)
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.decay = 0.99
        self.min_epsilon = 0.001
        self.model = RLAgent.model_builder(lrt=self.learning_rate)
        # TODO bla bla
        pass

    @staticmethod
    def get_possible_actions(board):
        tmp = np.reshape(board, (6, 7))
        for i in tmp:
            for j in i:
                print("{} ".format(j), end="")
            print()
        res = []
        return res

    @staticmethod
    def model_builder(input_size=7 * 6, action_size=7, lrt=0.008):
        # TODO improve the model
        model = Sequential()
        model.add(Dense(24, input_dim=input_size, activation='relu'))
        model.add(Dense(24, activation='relu'))

        # TODO make sure to use the right activation function for output layer
        model.add(Dense(action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=lrt))  # TODO tune lr, check other loss functions
        return model

    def step(self, observation, configuration):
        self.tmp += 1
        action = 0
        return action

    def enhance(self, state, q_state):
        pass

    def learn(self):
        pass


"""
model = model_builder()
predicted = model.predict(t)
action = 0
observation, reward, done, info = trainer.step(action)
predicted[0][action] = -100 + 0.8 * np.amax(model.predict(np.reshape(observation.board, [1, 7 * 6]))[0])
print(predicted)
model.fit(t, predicted, epochs=1)

"""