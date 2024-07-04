import numpy as np

class Actions:
    def __init__(self, env):
        self.env = env

    def choose_action(self, state, q_table, epsilon):
        ''' Given a state, choose an action. '''
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(list(q_table[state].values()))
