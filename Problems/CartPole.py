import gym
from Model.Agent import Agent

class CartPole:   
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        #self.env.seed(21)
        self.agent = Agent(self.env)
