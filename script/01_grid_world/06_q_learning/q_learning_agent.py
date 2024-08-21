import numpy as np
import random
from environment import Env
from collections import defaultdict

class QLearningAgent:
    def __init__(self, actions):
        # action = [0, 1, 2, 3] in order up, down, left, right
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # Update queue function from <s, a, r, s'> samples
    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        # Update the queue function using Bellman's Optimality Equation
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

    # return an action based on the e-greedy policy based on the q function
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # return random behavior
            action = np.random.choice(self.actions)
        else:
            # Return an action based on a q function
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()

        while True:
            env.render()
            env.title(f'Q Learning - Episode {episode}')

            # Select an action for the current state
            action = agent.get_action(str(state))
            # Get the next state after taking an action, whether the reward episode is over or not
            next_state, reward, done = env.step(action)

            # Update the q function with <s,a,r,s'>.
            agent.learn(str(state), action, reward, str(next_state))
            state = next_state
            # display all q functions on the screen
            env.print_value_all(agent.q_table)

            if done:
                break
