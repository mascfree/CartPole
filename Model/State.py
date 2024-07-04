import numpy as np

class State:
    def __init__(self, env):
        self.env = env

    def build_state(self, features):
        ''' Build state by concatenating features (bins) into a single integer. '''
        try:
            return int("".join(map(str, features)))
        except Exception as e:
            print(f"Error in build_state: {e}")
            return None

    def discretize(self, obs):
        ''' Discretize the continuous observation space. '''
        try:
            cart_position_bins = np.linspace(-4.8, 4.8, 11)[1:-1]
            pole_angle_bins = np.linspace(-24, 24, 11)[1:-1]
            cart_velocity_bins = np.linspace(-10, 10, 11)[1:-1]
            angle_rate_bins = np.linspace(-10, 10, 11)[1:-1]

            state = [
                np.digitize(obs[0], bins=cart_position_bins),
                np.digitize(obs[1], bins=pole_angle_bins),
                np.digitize(obs[2], bins=cart_velocity_bins),
                np.digitize(obs[3], bins=angle_rate_bins)
            ]

            return state
        except Exception as e:
            print(f"Error in discretize: {e}")
            return None

    def create(self, obs):
        ''' Create state variable from observation. '''
        try:
            discretized_obs = self.discretize(obs)
            if discretized_obs is not None:
                state = self.build_state(discretized_obs)
                return state
            else:
                return None
        except Exception as e:
            print(f"Error in create: {e}")
            return None
