#Source: https://towardsdatascience.com/why-going-from-implementing-q-learning-to-deep-q-learning-can-be-difficult-36e7ea1648af
from collections import deque

import numpy as np
import tensorflow as tf
from Algorithms.help_functions import *


def dense(x, weights, bias, activation=tf.identity, **activation_kwargs):
    """Dense layer."""
    z = tf.matmul(x, weights) + bias
    return activation(z, **activation_kwargs)


def init_weights(shape, initializer):
    """Initialize weights for tensorflow layer."""
    weights = tf.Variable(
        initializer(shape),
        trainable=True,
        dtype=tf.float32
    )

    return weights


class Network(object):
    """Q-function approximator."""

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size=[16, 8],
                 weights_initializer=tf.initializers.glorot_uniform(),
                 bias_initializer=tf.initializers.zeros(),
                 optimizer=tf.optimizers.Adam,
                 **optimizer_kwargs):
        """Initialize weights and hyperparameters."""
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        np.random.seed(41)

        self.initialize_weights(weights_initializer, bias_initializer)
        self.optimizer = optimizer(**optimizer_kwargs)

    def initialize_weights(self, weights_initializer, bias_initializer):
        """Initialize and store weights."""
        wshapes = [[self.input_size, self.hidden_size[0]]]
        layer_indices = np.arange(1, len(self.hidden_size))
        for i in layer_indices:
            wshapes.append([self.hidden_size[i-1], self.hidden_size[i]])
        wshapes.append([self.hidden_size[-1], self.output_size])
        '''
        wshapes2 = [
            [self.input_size, self.hidden_size[0]],
            [self.hidden_size[0], self.hidden_size[1]],
            [self.hidden_size[1], self.output_size]
        ]
        '''
        bshapes = []
        bias_indices = np.arange(0, len(self.hidden_size))
        for i in bias_indices:
            bshapes.append([1, self.hidden_size[i]])
        bshapes.append([1, self.output_size])
        '''
        bshapes2 = [
            [1, self.hidden_size[0]],
            [1, self.hidden_size[1]],
            [1, self.output_size]
        ]
        '''

        self.weights = [init_weights(s, weights_initializer) for s in wshapes]
        self.biases = [init_weights(s, bias_initializer) for s in bshapes]

        self.trainable_variables = self.weights + self.biases

    def model(self, inputs):
        """Given a state vector, return the Q values of actions."""

        h = dense(inputs, self.weights[0], self.biases[0], tf.nn.relu)
        h_s = [h]
        h_indices = np.arange(1, len(self.hidden_size))
        for i in h_indices:
            h = dense(h_s[i-1], self.weights[i], self.biases[i], tf.nn.relu)
            h_s.append(h)
        out = dense(h_s[-1], self.weights[-1], self.biases[-1])
        '''
        h1 = dense(inputs, self.weights[0], self.biases[0], tf.nn.relu)
        h2 = dense(h1, self.weights[1], self.biases[1], tf.nn.relu)

        out2 = dense(h2, self.weights[2], self.biases[2])
        '''

        return out

    def train_step(self, inputs, targets, actions_one_hot):
        """Update weights."""
        with tf.GradientTape() as tape:
            qvalues = tf.squeeze(self.model(inputs))
            preds = tf.reduce_sum(qvalues * actions_one_hot, axis=1)
            loss = tf.losses.mean_squared_error(targets, preds)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))


class Memory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False
        )

        return [self.buffer[i] for i in index]

    def __len__(self):
        """Interface to access buffer length."""
        return len(self.buffer)


class DeepQLearning(object):
    """Deep Q-learning agent."""

    def __init__(self,
                 n_x,
                 n_a,
                 n_y,
                 data,
                 constraint,
                 target_update_freq=10,
                 discount=0.99,
                 batch_size=32,
                 max_explore=1,
                 min_explore=0.05,
                 anneal_rate=(1 / 100000),
                 replay_memory_size=100,
                 replay_start_size=64):
        """Set parameters, initialize network."""
        action_space_size = n_a + 1
        state_space_size = n_x + n_a
        self.action_space_size = action_space_size

        self.online_network = Network(state_space_size, action_space_size)
        self.target_network = Network(state_space_size, action_space_size)

        self.update_target_network()

        # training parameters
        self.target_update_freq = target_update_freq
        self.discount = discount
        self.batch_size = batch_size

        # policy during learning
        self.max_explore = max_explore + (anneal_rate * replay_start_size)
        self.min_explore = min_explore
        self.anneal_rate = anneal_rate
        self.steps = 0

        # replay memory
        self.memory = Memory(replay_memory_size)
        self.replay_start_size = replay_start_size
        self.experience_replay = Memory(replay_memory_size)

        self.name = 'Deep Q-learning'
        self.label = 'DQL'
        self.n_a = n_a
        self.stop_action = n_a
        self.n_x = n_x
        self.n_y = n_y
        self.data = data
        self.constraint = constraint

    def handle_episode_start(self):
        self.last_state, self.last_action = None, None

    def step(self, state, last_reward, last_state, last_action, training=True, forbidden_actions=()):
        """Observe state and rewards, select action.

        It is assumed that `observation` will be an object with
        a `state` vector and a `reward` float or integer. The reward
        corresponds to the action taken in the previous step.
        """

        if training:
            self.steps += 1

            if last_state is not None:
                experience = {
                    "state": last_state,
                    "action": last_action,
                    "reward": last_reward,
                    "next_state": state
                }

                self.memory.add(experience)

            if self.steps > self.replay_start_size:
                self.train_network()

                if self.steps % self.target_update_freq == 0:
                    self.update_target_network()

        action = self.policy(state, training, forbidden_actions)

        return action

    def policy(self, state, training, forbidden_actions=()):
        """Epsilon-greedy policy for training, greedy policy otherwise."""
        explore_prob = self.max_explore - (self.steps * self.anneal_rate)
        explore = False #max(explore_prob, self.min_explore) > np.random.rand()

        if training and explore:
            if len(forbidden_actions) > 0:
                possible_actions = np.arange(self.action_space_size)
                probabilities = np.zeros(len(possible_actions))
                for i, is_forbidden in enumerate(forbidden_actions):
                    if not is_forbidden:
                        probabilities[i] = 1
                probabilities /= np.sum(probabilities)
                action = np.random.choice(possible_actions, 1, p=probabilities)[0]
            else:
                action = np.random.randint(self.action_space_size)
        else:
            inputs = np.expand_dims(state, 0)
            qvalues = self.online_network.model(inputs)
            temp_qvalues = list(qvalues[0])
            temp_h_state = state[self.n_x:]
            for i in range(len(temp_h_state)):
                if temp_h_state[i] != -1:
                    temp_qvalues[i] = -np.inf
            for i, is_forbidden in enumerate(forbidden_actions):
                if is_forbidden:
                    temp_qvalues[i] = -np.inf
            action = np.squeeze(np.argmax(temp_qvalues))
            assert np.max(temp_qvalues) != -np.inf
        return action

    def update_target_network(self):
        """Update target network weights with current online network values."""
        variables = self.online_network.trainable_variables
        variables_copy = [tf.Variable(v) for v in variables]
        self.target_network.trainable_variables = variables_copy

    def train_network(self):
        """Update online network weights."""
        batch = self.memory.sample(self.batch_size)
        inputs = np.array([b["state"] for b in batch])
        actions = np.array([b["action"] for b in batch])
        rewards = np.array([b["reward"] for b in batch])
        next_inputs = np.array([b["next_state"] for b in batch])

        actions_one_hot = np.eye(self.action_space_size)[actions]

        next_qvalues = np.squeeze(self.target_network.model(next_inputs))
        targets = rewards + self.discount * np.amax(next_qvalues, axis=-1)

        self.online_network.train_step(inputs, targets, actions_one_hot)

    def learn(self):
        x_s = self.data['x']
        histories = self.data['h']
        n_patients = len(x_s)
        n_batch_trainings = 0
        self.handle_episode_start()
        patient_indices = np.arange(n_patients)
        np.random.shuffle(patient_indices)
        for j in patient_indices:
            x = x_s[j]
            history = histories[j]

            last_action, outcome = history[-1]
            h_state = history_to_state(history[:-1], self.n_a)
            last_reward = self.get_reward(last_action, h_state, x)
            state = np.concatenate((x, h_state)).astype('float32')
            next_state = state.copy()
            next_state[last_action] = outcome
            self.step(next_state, last_reward, state, last_action, training=True)

            last_action = self.stop_action
            h_state = history_to_state(history, self.n_a)
            last_reward = self.get_reward(last_action, h_state, x)
            next_state = state.copy()
            next_state[last_action] = outcome
            self.step(next_state, last_reward, state, last_action, training=True)

        for i in range(n_batch_trainings):
            self.train_network()
            self.update_target_network()

    def evaluate(self, patient):
        z, x, y_fac = patient
        y = np.array([-1] * self.n_a)
        history = []
        state = np.concatenate((x, y)).astype('float32')
        forbidden_actions = (y_fac == -1)
        forbidden_actions_init = np.concatenate((forbidden_actions, np.array([True])))
        action = self.policy(state, training=False, forbidden_actions=forbidden_actions_init)
        while action != self.stop_action and len(history) < self.n_a:
            y[action] = int(y_fac[action])
            history.append([action, y[action]])
            if y[action] == self.n_y-1:
                break
            state = np.concatenate((x, y)).astype('float32')
            action = self.policy(state, training=False, forbidden_actions=forbidden_actions)
        return history

    def get_reward(self, action, history, x):
        gamma = self.constraint.no_better_treatment_exist(history, x)
        if action == self.stop_action and gamma == 0:
            return -10000000
        elif action == self.stop_action and gamma == 1:
            return 0
        elif self.stop_action > action >= 0:
            return -1
        else:
            import sys
            print(gamma, action, history)
            sys.exit()

    def normalize_data(self, input):
        #Normalization seem to suck for this input
        #input = (input - -1)/(self.n_y - -1)
        return input


