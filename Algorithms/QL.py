import numpy as np

class QL:
    def __init__(self, agent, alpha=0.1, epsilon=1.0, gamma=0.9, epsilon_decay=0.99, epsilon_min=0.01):
        self.agent = agent
        self.alpha = alpha # tasa de aprendizaje
        self.epsilon = epsilon # Tasa de exploración 
        self.gamma = gamma # Factor de descuento
        self.epsilon_decay = epsilon_decay 
        self.epsilon_min = epsilon_min
        self.q_table = dict()  

    def learn(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in range(self.agent.env.action_space.n)}
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in range(self.agent.env.action_space.n)}
        
        max_Q_next_state = max(self.q_table[next_state].values()) if next_state in self.q_table else 0
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_Q_next_state - self.q_table[state][action])

    def train(self, episodes):
        for _ in range(episodes):
            current_state = self.agent.state.create(self.agent.env.reset())
            if current_state not in self.q_table:
                self.q_table[current_state] = {a: 0.0 for a in range(self.agent.env.action_space.n)}
            done = False
            while not done:
                action = self.agent.action_chooser.choose_action(current_state, self.q_table, self.epsilon)
                obs, reward, done, _ = self.agent.env.step(action)
                next_state = self.agent.state.create(obs)
                if next_state not in self.q_table:
                    self.q_table[next_state] = {a: 0.0 for a in range(self.agent.env.action_space.n)}
                self.learn(current_state, action, reward, next_state)
                current_state = next_state
                if done:
                    break
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def test(self, episodes):
        total_rewards = []
        for _ in range(episodes):
            current_state = self.agent.state.create(self.agent.env.reset())
            done = False
            total_reward = 0
            while not done:
                action = np.argmax(list(self.q_table[current_state].values()))
                obs, reward, done, _ = self.agent.env.step(action)
                current_state = self.agent.state.create(obs)
                total_reward += reward
                if done:
                    total_rewards.append(total_reward)
                    break
        return total_rewards
