from Algorithms.probability_approximator import ProbabilityApproximator
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from  Algorithms.help_functions import *


class FunctionApproximation(ProbabilityApproximator):
    def __init__(self, n_x, n_a, n_y, data):
        super().__init__(n_x, n_a, n_y, data)
        #self.model = SVC(C=1, kernel='poly', degree=7, probability=True)
        self.model = RandomForestClassifier(n_jobs=-1)
        self.name = 'Random forest approximator'
        self.xs = data['x']
        self.histories = data['h']
        self.n_samples = len(self.xs)
        self.n_features = n_x + n_a + n_a + 1
        self.predictions_dict = {}
        input_data = np.zeros((self.n_samples, self.n_features))
        output_data = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            x = self.xs[i]
            h = self.histories[i]
            inp, outp = self.get_features(x, h)
            input_data[i] = inp
            output_data[i] = outp

        self.model.fit(input_data, output_data)

    def get_features(self, x, history):
        last_action, last_outcome = history[-1]
        actions, outcomes = self.history_to_actions_and_outcomes(history[:-1])
        features = self.fill_feature_vector(x, actions, outcomes, last_action)
        return features, last_outcome

    def history_to_actions_and_outcomes(self, history):
        actions = np.zeros(self.n_a)
        outcomes = np.zeros(self.n_a)
        for intervention in history:
            treatment, outcome = intervention
            actions[treatment] = 1
            outcomes[treatment] = outcome
        return actions, outcomes

    def state_to_actions_and_outcomes(self, state):
        actions = np.zeros(self.n_a)
        outcomes = np.zeros(self.n_a)
        for treatment, outcome in enumerate(state):
            if outcome > -1:
                actions[treatment] = 1
                outcomes[treatment] = outcome
        return actions, outcomes

    def fill_feature_vector(self, x, actions, outcomes, last_action=None):
        if last_action is None:
            n_features = self.n_features-1
        else:
            n_features = self.n_features
        features = np.zeros(n_features)
        features[:self.n_x] = x
        features[self.n_x:self.n_x + self.n_a] = actions
        features[self.n_x + self.n_a:self.n_x + self.n_a + self.n_a] = outcomes
        if last_action is not None:
            features[self.n_x + self.n_a + self.n_a:self.n_x + self.n_a + self.n_a + 1] = last_action
        return features

    def prepare_calculation(self, x, history, action=None):
        if action is None:
            probability_of_outcome_approximation = np.zeros((self.n_a, self.n_y))
            for action in range(self.n_a):
                if history[action] == -1:
                    old_actions, old_outcomes = self.state_to_actions_and_outcomes(history)
                    features = self.fill_feature_vector(x, old_actions, old_outcomes, action)
                    h = hash_array(features)
                    if h in self.predictions_dict:
                        prediction = self.predictions_dict[h]
                    else:
                        prediction = self.model.predict_proba(features.reshape(1, -1))[0]
                        self.predictions_dict[h] = prediction
                    probability_of_outcome_approximation[action] = prediction
        else:
            old_actions, old_outcomes = self.state_to_actions_and_outcomes(history)
            features = self.fill_feature_vector(x, old_actions, old_outcomes, action)
            probability_of_outcome_approximation = self.model.predict_proba(features.reshape(1, -1))
        return probability_of_outcome_approximation

    def calculate_probability(self, probability_of_outcome_approximation, outcome):
        return probability_of_outcome_approximation[0][outcome]

    def calculate_probability_greedy(self, state, best_outcome):
        x, history = state
        probability_of_outcome_approximation = self.prepare_calculation(x, history)
        return super(FunctionApproximation, self).calculate_probability_greedy(probability_of_outcome_approximation, best_outcome)