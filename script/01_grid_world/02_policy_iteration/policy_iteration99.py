import random
from environment99 import GraphicDisplay, Env


class PolicyIteration:
    def __init__(self, env):
        # Instantiate the environment
        self.env = env
        # Table for the value function
        self.value_table = [[0.0] * env.width for _ in range(env.height)]
        # Table for the policy with uniform distribution of all actions
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]] * env.width
                                    for _ in range(env.height)]
        # Terminal state
        self.policy_table[4][4] = []
        # Discount factor
        self.discount_factor = 0.9

    def policy_evaluation(self):

        # Initialize the next value table with zeros
        next_value_table = [[0.00] * self.env.width
                                    for _ in range(self.env.height)]

        # Update the value function for all states
        for state in self.env.get_all_states():
            value = 0.0
            # Do not update the value function for the terminal state
            if state == [4, 4]:
                next_value_table[state[0]][state[1]] = value
                continue

            # Update the value function via Bellman expectation equation
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += (self.get_policy(state)[action] *
                          (reward + self.discount_factor * next_value))

            next_value_table[state[0]][state[1]] = round(value, 2)

        self.value_table = next_value_table

    # Update the policy
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [4, 4]:
                continue
            value = -99999
            max_index = []
            # Initialize the next policy with zeros
            result = [0.0, 0.0, 0.0, 0.0]

            # Find the best action which maximizes the value function
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                # Calculate the Q function (i.e., value) for the state and action
                temp = reward + self.discount_factor * next_value

                # Update the best action
                # If the value is the same as the maximum value, append the action (i.e., multiple actions might be the best)
                if temp == value:
                    max_index.append(index)
                elif temp > value:
                    value = temp
                    max_index.clear()
                    max_index.append(index)

            # Only the best policy has non-zero value
            prob = 1 / len(max_index)
            for index in max_index:
                result[index] = prob

            # Update the policy with the best action for all states
            next_policy[state[0]][state[1]] = result

        # Return the updated policy
        self.policy_table = next_policy

    # Get the best action from the policy at a given state
    def get_action(self, state):
        # Randomly choose the action according to the policy if there are multiple best actions
        random_pick = random.randrange(100) / 100

        policy = self.get_policy(state)
        policy_sum = 0.0
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index

    # Get the policy at a given state
    def get_policy(self, state):
        if state == [4, 4]:
            return 0.0
        return self.policy_table[state[0]][state[1]]

    # Get the value of the value function at a given state
    def get_value(self, state):
        # Round the value function to 2 decimal places
        return round(self.value_table[state[0]][state[1]], 2)

if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()
