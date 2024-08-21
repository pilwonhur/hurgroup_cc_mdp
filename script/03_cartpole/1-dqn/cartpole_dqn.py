import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 300


# DQN agent in the cartpole example
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        # Define the size of states and actions
        self.state_size = state_size
        self.action_size = action_size

        # DQN hyperparameter
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # Replay memory, max size 2000
        self.memory = deque(maxlen=2000)

        # Create a model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Initialize the target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_dqn_trained.h5")

    # Create a neural network with state as input and q function as output
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    # Update the target model with the model's weights
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Select behaviors with the e-Greed policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s, a, r, s'> to replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Train a model with a randomized batch from replay memory
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # sample randomly from memory by the batch size
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # q function in the model for the current state
        # q functions in the target model for the following states
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # Update targets using Bellman's Optimization Equation
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":
    # CartPole-v1 environment, maximum number of timesteps 500
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create a DQN agent
    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        # initialize env
        state = env.reset()
        state = np.reshape(state[0], [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # Select an action in the current state
            action = agent.get_action(state)
            # Advance one timestep in the environment with the selected behavior
            next_state, reward, done, info, truncated = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # -100 reward if episode ends midway through
            reward = reward if not done or score == 499 else -100

            # Save sample <s, a, r, s'> to replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # Learn every timestep
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward
            state = next_state

            if done:
                # Update the target model with the model's weights for each episode
                agent.update_target_model()

                score = score if score == 500 else score + 100
                # Output learning outcomes per episode
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

                # Stop learning if the score average of the previous 10 episodes is greater than 490
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.model.save_weights("./save_model/cartpole_dqn.weights.h5")
                    sys.exit()
