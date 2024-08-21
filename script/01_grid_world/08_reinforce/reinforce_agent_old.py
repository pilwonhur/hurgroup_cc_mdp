import copy
import pylab
import numpy as np
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
import tensorflow as tf

EPISODES = 2500

# REINFORCE agent in the Gridworld example
class ReinforceAgent:
    def __init__(self):
        self.load_model = False
        # Define all possible behaviors
        self.action_space = [0, 1, 2, 3, 4]
        # Define the size of states and actions
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99 
        self.learning_rate = 0.001

        self.model = self.build_model()
        dummy_input = np.zeros((1, self.state_size))
        # self.model(dummy_input)
        self.model.predict(dummy_input)
        print(self.model.get_weights())
        # model(tf.keras.Input((810, 1440, 3)))
        # depth = model.get_output_shape_at(0)
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights('./save_model/reinforce_trained.h5')
    
    # Create an artificial neural network with state as input and probability of each behavior as output
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model
    
    # Create error and training functions to update the policy neural network
    def optimizer(self):
        action = tf.keras.Input(shape=[5], dtype=tf.float32)
        discounted_rewards = tf.keras.Input(shape=[None, ], dtype=tf.float32)
        
        # Compute the cross-entropy error function
        action_prob = tf.reduce_sum(action * self.model.output, axis=1)
        cross_entropy = tf.math.log(action_prob) * discounted_rewards
        loss = -tf.reduce_sum(cross_entropy)
        
        # Create a training function to update the policy neural network
        optimizer = Adam(learning_rate=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights,[],
                                        loss)
        train = K.function([self.model.input, action, discounted_rewards], [],
                           updates=updates)

        return train

    # Selecting behaviors with a policy neural network
    def get_action(self, state):
        policy = self.model.predict(state)[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]
    
    # Calculate the return value
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    
    # Save states, behaviors, and rewards for a single episode
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # Update the policy neural network
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    # Create an environment and agent
    env = Env()
    agent = ReinforceAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        # initialize env
        state = env.reset()
        state = np.reshape(state, [1, 15])

        while not done:
            global_step += 1
            # Select an action for the current state
            action = agent.get_action(state)
            # Collect a sample after one timestep in the environment with the selected behavior
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])

            agent.append_sample(state, action, reward)
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                # Update the policy neural network every episode
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                score = round(score,2)
                print("episode:", e, "  score:", score, "  time_step:",
                      global_step)

        # Save learning outcome output and model every 100 episodes
        if e % 100 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/reinforce.png")
            agent.model.save_weights("./save_model/reinforce.h5")
