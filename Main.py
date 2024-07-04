from Problems.CartPole import CartPole
from Algorithms.QL import QL
import numpy as np

def main():
    # Create a cartpole environment
    env = CartPole()
    
    # Initialize Q-Learning
    ql = QL(env.agent)
    
    # Train the agent
    ql.train(1000)
    
    # Test the agent
    testing_totals = ql.test(100)
    
    # Display statistics
    if np.mean(testing_totals) >= 195.0:
        print("Environment SOLVED!!!")
    else:
        print("Environment not solved. Must get an average reward of 195.0 or greater for 100 consecutive trials.")

if __name__ == "__main__":
    main()
