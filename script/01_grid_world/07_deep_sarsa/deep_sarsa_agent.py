import copy
import pylab
import random
import numpy as np
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 1000


# DeepSalsa Agent in the Gridworld example
class DeepSARSAgent:
    def __init__(self):
        self.load_model = False
        # Define all possible behaviors for the agent
        self.action_space = [0, 1, 2, 3, 4]
        # Define the size of a state and the size of an action
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = self.build_model()

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights('./save_model/deep_sarsa_trained.h5')

    # Create a neural network whose state is the output of the input q function
    def build_model(self):
        model = Sequential()
        model.add(Dense(30, input_dim=self.state_size, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # Select an action with the e-Greed method
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # return random behavior
            return random.randrange(self.action_size)
        else:
            # Calculate behaviors from models
            state = np.float32(state)
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        # Update Salsa's q function expression
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor *
                              self.model.predict(next_state)[0][next_action])

        # output value reshape
        target = np.reshape(target, [1, 5])
        # Update the neural network
        self.model.fit(state, target, epochs=1, verbose=0)


if __name__ == "__main__":
    # Create an environment and agent
    env = Env()
    agent = DeepSARSAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, 15])

        while not done:
            # initialize env
            global_step += 1

            # Select an action for the current state
            action = agent.get_action(state)
            # Collect a sample after one timestep in the environment with the selected behavior
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])
            next_action = agent.get_action(next_state)
            # Train the model with samples
            agent.train_model(state, action, reward, next_state, next_action,
                              done)
            state = next_state
            score += reward

            state = copy.deepcopy(next_state)

            if done:
                # Output learning outcomes per episode
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/deep_sarsa_.png")
                print("episode:", e, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon)

        # Save model every 100 episodes
        if e % 100 == 0:
            # agent.model.save_weights("./save_model/deep_sarsa.h5")
            agent.model.save_weights("./save_model/deep_sarsa.weights.h5")
