from kaggle_environments import evaluate, make, utils
import random
import gym
import numpy as np

import RLAgent as agt

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
trainer = env.train([None, "random"])
agent = agt.RLAgent()
for _ in range(1000):
    observation = trainer.reset()
    while not env.done:
        current_state_ = np.reshape(observation.board, [1, 7 * 6])
        my_action = agent.step(current_state_)
        next_state, reward, done, info = trainer.step(my_action)

        next_state_ = np.reshape(next_state.board, [1, 7 * 6])

        agent.enhance(current_state_, my_action, reward, next_state_, env.done)

        agent.learn()
        observation = next_state
