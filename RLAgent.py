import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import activations


class RLAgent:
    def __init__(self):
        self.counter = 0
        self.tmp = 0
        self.data_set = deque(maxlen=10000)
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.decay = 0.99
        self.min_epsilon = 0.01
        self.model = RLAgent.model_builder(lrt=self.learning_rate)
        self.batch_size = 100
        self.gamma = 0.1
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
    def model_builder(input_size=7 * 6, action_size=7, lrt=0.001):
        # TODO improve the model
        model = Sequential()
        model.add(Dense(24, input_dim=input_size, activation='relu'))
        model.add(Dense(24, activation='relu'))

        # TODO make sure to use the right activation function for output layer
        model.add(Dense(action_size, activation=activations.linear))
        model.compile(loss='mse', optimizer=Adam(lr=lrt))  # TODO tune lr, check other loss functions
        model.summary()
        return model

    def step(self, state):
        rand_tmp = np.random.random()
        if rand_tmp <= self.epsilon:
            return np.random.randint(0, 6)

        return np.argmax(self.model.predict(state)[0])

    def enhance(self, current_state, action, reward, next_state, is_final):
        data = (
            current_state,
            action,
            reward,
            next_state,
            is_final
        )
        self.data_set.append(data)

    def convertDS(self, minibatch):
        X = np.array([])
        Y = np.array([])
        c = 0
        for cur_state, action, reward, next_state, is_final in minibatch:
            c += 1
            # Q(S,a) = reward + gamma * argmax(Next_State)
            ret = self.model.predict(cur_state)[0]
            ret[action] = reward
            if not is_final:
                ret[action] += self.gamma * np.amax(self.model.predict(next_state)[0])
            X = np.append(X, cur_state)
            Y = np.append(Y, ret)

        return X.reshape((c, 42)), Y.reshape((c, 7))

    def learn(self):
        if len(self.data_set) < self.batch_size + 100:
            return
        self.counter += 1
        self.epsilon = min(self.epsilon, self.epsilon * self.decay)

        mini_batch = random.sample(self.data_set, self.batch_size)

        X, Y = self.convertDS(minibatch=mini_batch)
        print("Epoch : {}".format(self.counter))
        self.model.fit(X, Y, verbose=1)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


"""
model = model_builder()
predicted = model.predict(t)
action = 0
observation, reward, done, info = trainer.step(action)
predicted[0][action] = -100 + 0.8 * np.amax(model.predict(np.reshape(observation.board, [1, 7 * 6]))[0])
print(predicted)
model.fit(t, predicted, epochs=1)

"""
