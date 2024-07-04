import numpy as np
import random
from Model.State import State
from Model.Action import Actions

class Agent:
    def __init__(self, env):
        self.env = env
        self.state = State(env)
        self.action_chooser = Actions(env)
        self._set_seed()
        self.training_trials = 0
        self.testing_trials = 0

    def _set_seed(self):
        ''' Set random seeds for reproducibility. '''
        np.random.seed(21)
        random.seed(21)

    def create_Q(self, state, valid_actions, q_table):
        ''' Update the Q table given a new state/action pair. '''
        if state not in q_table:
            q_table[state] = {action: 0.0 for action in valid_actions}
