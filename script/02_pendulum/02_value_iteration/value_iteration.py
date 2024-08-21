# -*- coding: utf-8 -*-
from environment import GraphicDisplay, Env

class ValueIteration:
    def __init__(self, env):
        # Initialize environment
        self.env = env
        # Initialize value table with 0
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # Discount factor
        self.discount_factor = 0.9

    # Value iteration
    # Update value table using Bellman Optimality Equation
    def value_iteration(self):
        next_value_table = [[0.0] * self.env.width for _ in
                            range(self.env.height)]
        for state in self.env.get_all_states():
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = 0.0
                continue
            # Initialize value list
            value_list = []

            # Calculate value function for all actions
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value_list.append((reward + self.discount_factor * next_value))
            # Select the maximum value
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)
        self.value_table = next_value_table

    # Get action according to the current optimal value function
    def get_action(self, state):
        action_list = []
        max_value = -99999

        if state == [2, 2]:
            return []

        # Calculate the value of all actions (reward + gamma * next_value = same as q function) from the Bellman Optimality Equation
        # Select the action with the maximum value
        for action in self.env.possible_actions:

            next_state = self.env.state_after_action(state, action)
            reward = self.env.get_reward(state, action)
            next_value = self.get_value(next_state)
            value = (reward + self.discount_factor * next_value)

            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)

        return action_list

    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)

if __name__ == "__main__":
    env = Env()
    value_iteration = ValueIteration(env)
    grid_world = GraphicDisplay(value_iteration)
    grid_world.mainloop()
