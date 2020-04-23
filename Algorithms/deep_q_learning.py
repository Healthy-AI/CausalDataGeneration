#Source: https://towardsdatascience.com/why-going-from-implementing-q-learning-to-deep-q-learning-can-be-difficult-36e7ea1648af
from collections import deque

import numpy as np
import tensorflow as tf
from Algorithms.help_functions import *
from Algorithms.function_approximation import FunctionApproximation


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
        tf.random.set_seed(42)

        self.initialize_weights(weights_initializer, bias_initializer)
        self.optimizer = optimizer(**optimizer_kwargs)

    def initialize_weights(self, weights_initializer, bias_initializer):
        """Initialize and store weights."""
        wshapes = [[self.input_size, self.hidden_size[0]]]
        layer_indices = np.arange(1, len(self.hidden_size))
        for i in layer_indices:
            wshapes.append([self.hidden_size[i-1], self.hidden_size[i]])
        wshapes.append([self.hidden_size[-1], self.output_size])

        bshapes = []
        bias_indices = np.arange(0, len(self.hidden_size))
        for i in bias_indices:
            bshapes.append([1, self.hidden_size[i]])
        bshapes.append([1, self.output_size])

        self.weights = [init_weights(s, weights_initializer) for s in wshapes]
        self.biases = [init_weights(s, bias_initializer) for s in bshapes]

        self.trainable_variables = self.weights + self.biases

    def model(self, inputs):
        """Given a state vector, return the Q values of actions."""

        h_s = [dense(inputs, self.weights[0], self.biases[0], tf.nn.relu)]
        h_indices = np.arange(1, len(self.hidden_size))
        for i in h_indices:
            h = dense(h_s[i-1], self.weights[i], self.biases[i], tf.nn.relu)
            h_s.append(h)
        out = dense(h_s[-1], self.weights[-1], self.biases[-1])

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

    def __init__(self):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque()

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
                 approximator,
                 target_update_freq=1000,
                 discount=1,
                 batch_size=16,
                 n_batch_trainings=10000):
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

        self.name = 'Constrained Deep Q-learning'
        self.label = 'CDQL'
        self.n_a = n_a
        self.stop_action = n_a
        self.n_x = n_x
        self.n_y = n_y
        self.max_outcome = n_y - 1
        data_split_len = int(len(data['x'])*0.8)
        self.approximator = approximator
        self.data = {'x': data['x'][:data_split_len], 'h': data['h'][:data_split_len]}
        self.validation_data = data
        self.validation_data['x'] = data['x'][data_split_len:]
        self.validation_data['h'] = data['h'][data_split_len:]
        self.constraint = constraint
        self.n_batch_trainings = n_batch_trainings
        self.batch_size = batch_size

        # replay memory
        self.memory = Memory()

        self.validation_data = self.training_to_validation_data(self.validation_data)

    def add_to_memory(self, state, last_reward, last_state, last_action):
        """Observe state and rewards, select action.

        It is assumed that `observation` will be an object with
        a `state` vector and a `reward` float or integer. The reward
        corresponds to the action taken in the previous step.
        """

        experience = {
            "state": last_state,
            "action": last_action,
            "reward": last_reward,
            "next_state": state
        }
        self.memory.add(experience)

    def policy(self, state, forbidden_actions=()):

        inputs = np.expand_dims(state, 0)
        qvalues = self.target_network.model(inputs)
        temp_qvalues = list(qvalues[0])
        temp_h_state = state[self.n_x:]
        for i in range(len(temp_h_state)):
            if temp_h_state[i] != -1:
                temp_qvalues[i] = -np.inf
        for i, is_forbidden in enumerate(forbidden_actions):
            if is_forbidden:
                temp_qvalues[i] = -np.inf
        action = np.squeeze(np.argmax(temp_qvalues))
        #assert np.max(temp_qvalues) != -np.inf
        return action

    def update_target_network(self):
        """Update target network weights with current online network values."""
        variables = self.online_network.trainable_variables
        variables_copy = [tf.Variable(v) for v in variables]
        self.target_network.trainable_variables = variables_copy
        self.set_target_variables(variables_copy)

    def set_target_variables(self, variables):
        self.target_network.weights = variables[:int(len(variables) / 2)]
        self.target_network.biases = variables[int(len(variables) / 2):]

    def train_network(self):
        """Update online network weights."""
        batch = self.memory.sample(self.batch_size)
        inputs = np.array([b["state"] for b in batch])
        actions = np.array([b["action"] for b in batch])
        rewards = np.array([b["reward"] for b in batch])
        next_inputs = np.array([b["next_state"] for b in batch])

        actions_one_hot = np.eye(self.action_space_size)[actions]

        next_qvalues = np.squeeze(self.target_network.model(next_inputs))
        target_actions = np.argmax(next_qvalues, axis=-1)
        target_actions_one_hot = np.eye(self.action_space_size)[target_actions]
        online_q_values = np.squeeze(self.online_network.model(next_inputs))
        targets = rewards + self.discount * tf.reduce_sum(online_q_values * target_actions_one_hot, axis=1) #np.amax(next_qvalues, axis=-1)

        self.online_network.train_step(inputs, targets, actions_one_hot)

    def learn(self):
        x_s = self.data['x']
        histories = self.data['h']
        n_interventions = len(x_s)
        print("Adding {} interventions to memory".format(n_interventions*2))
        for i in range(n_interventions):
            x = x_s[i]
            history = histories[i]

            last_action, outcome = history[-1]
            h_state = history_to_state(history[:-1], self.n_a)
            last_reward = self.get_reward(last_action, h_state, x)
            last_state = self.create_state(x, h_state)
            state = last_state.copy()
            state[self.action_index(last_action)] = outcome
            self.add_to_memory(state, last_reward, last_state, last_action)

            last_action = self.stop_action
            h_state = history_to_state(history, self.n_a)
            last_reward = self.get_reward(last_action, h_state, x)
            last_state = state
            self.add_to_memory(state, last_reward, last_state, last_action)

        print('Performing training')
        max_treatment_effect = 0
        min_search_time = self.n_a
        best_variables_copy = None
        for i in range(self.n_batch_trainings+1):
            self.train_network()
            if i % self.target_update_freq == 0:
                self.update_target_network()
                mean_treatment_effect, mean_num_tests = self.evaluate_validation()
                if mean_treatment_effect >= max_treatment_effect:
                    if mean_num_tests < min_search_time or mean_treatment_effect > max_treatment_effect:
                        max_treatment_effect = mean_treatment_effect
                        min_search_time = mean_num_tests
                        best_variables = self.target_network.trainable_variables
                        best_variables_copy = [tf.Variable(v) for v in best_variables]
                print('DQN performance: mean treatment effect {}, mean search time {}'.format(mean_treatment_effect, mean_num_tests))
        self.target_network.trainable_variables = best_variables_copy
        self.set_target_variables(best_variables_copy)
        print('Best time:', min_search_time)

    def evaluate(self, patient):
        z, x, y_fac = patient
        y = np.array([-1] * self.n_a)
        history = []
        state = self.create_state(x, y)
        forbidden_actions = (y_fac == -1)
        forbidden_actions_init = np.concatenate((forbidden_actions, [True]))
        action = self.policy(state, forbidden_actions=forbidden_actions_init)
        while action != self.stop_action and len(history) < self.n_a:
            y[action] = y_fac[action]
            history.append([action, y[action]])
            if y[action] == self.max_outcome:
                break
            state = self.create_state(x, y)
            action = self.policy(state, forbidden_actions=forbidden_actions)
        return history

    def evaluate_validation(self):
        mean_num_tests = 0
        max_treatment_effect = 0
        n_test_samples = len(self.validation_data)
        for patient in self.validation_data:
            history = self.evaluate(patient)
            mean_num_tests += len(history)
            best_found = np.max([intervention[1] for intervention in history])
            max_treatment_effect += best_found
        max_treatment_effect /= n_test_samples
        mean_num_tests /= n_test_samples
        return max_treatment_effect, mean_num_tests

    def training_to_validation_data(self, data):
        x_s = data['x']
        histories = data['h']
        patients = []
        for i in range(len(x_s)):
            x = x_s[i]
            history = histories[i]
            y_fac = np.ones(self.n_a)*-1
            for intervention in history:
                treatment, outcome = intervention
                y_fac[treatment] = outcome
            for treatment in range(len(y_fac)):
                if y_fac[treatment] == -1:
                    # estimate outcome
                    probs = self.approximator.calculate_probabilities(x, history_to_state(history, self.n_a), treatment)
                    estimated_outcome = np.random.choice(self.n_y, 1, p=probs)[0]
                    y_fac[treatment] = estimated_outcome
            z = -1
            patient = (z, x, y_fac)
            patients.append(patient)
        return patients


    def get_reward(self, action, history, x):
        gamma = self.constraint.no_better_treatment_exist(history, x)
        if action == self.stop_action and gamma == 0:
            return -100000000000
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

    def create_state(self, x, h_state):
        state = np.concatenate((x, h_state)).astype('float32')
        return state

    def action_index(self, action_index):
        return self.n_x + action_index
