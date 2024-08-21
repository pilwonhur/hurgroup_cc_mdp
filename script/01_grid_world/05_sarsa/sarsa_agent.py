import numpy as np
import random
from collections import defaultdict
from environment import Env


class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.05
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # Update the q function from samples in <s, a, r, s', a'>.
    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        new_q = (current_q + self.learning_rate *
                (reward + self.discount_factor * next_state_q - current_q))
        self.q_table[state][action] = new_q

    # return behavior according to the e-greed policy
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
    agent = SARSAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        # Initialize the game environment and state
        state = env.reset()
        # Select an action for the current state
        action = agent.get_action(str(state))
        
        env.title(f'SARSA - Episode {episode}')
        
        while True:
            env.render()

            # Get whether the next state reward episode for an action has ended
            next_state, reward, done = env.step(action)
            # Select the next action in the next state
            next_action = agent.get_action(str(next_state))

            # Update the q function with <s,a,r,s',a'>.
            agent.learn(str(state), action, reward, str(next_state), next_action)

            state = next_state
            action = next_action

            # display all q functions on the screen
            env.print_value_all(agent.q_table)

            if done:
                break

