import psycopg2
import numpy as np
import base64
import random


class AntibioticsDatabase:
    def __init__(self, seed=None):
        self.random = np.random.RandomState()
        self.random.seed(seed)
        self.antibiotic_to_treatment_dict = {}
        self.antibiotic_counter = 0
        self.n_x = 2
        self.x_counter = 0
        self.organism_to_x_dict = {}
        self.n_training_samples = None
        self.antibiotics_training_data = None
        self.antibiotics_test_data = None
        self.name = 'Antibiotics'
        self.antibiotic_limit = 6
        self.n_a = None
        self.n_y = 3

    def get_n_a(self):
        return min(self.antibiotic_counter, self.antibiotic_limit)

    def get_data(self):
        pw = base64.b64decode(b'aGVhbHRoeUFJ').decode("utf-8")
        conn = psycopg2.connect(database="mimic", user="postgres", password=pw)
        cur = conn.cursor()
        cur.execute('set search_path=mimiciii')
        cur.execute('SELECT subject_id, org_name, ab_name, interpretation FROM microbiologyevents WHERE ab_name IS NOT NULL;')
        data = cur.fetchall()
        patients = {}
        self.random.shuffle(data)
        for chartevent in data:
            subject_id = chartevent[0]
            organism = chartevent[1]
            treatment = self.antibiotic_to_treatment(chartevent[2])
            outcome = interpretation_to_outcome(chartevent[3])
            if not self.too_many_x(organism) and treatment < self.antibiotic_limit and outcome is not None:
                intervention = np.array([treatment, outcome])
                try:
                    if treatment not in [intervention[0] for intervention in patients[subject_id][organism]]:
                        patients[subject_id][organism].append(intervention)
                except KeyError:
                    patients[subject_id] = {organism: [intervention]}

        antibiotics_data = {'z': [], 'x': [], 'h': []}
        for subject_id, data in patients.items():
            for organism, history in data.items():
                x = self.organism_to_x_dict[organism]
                antibiotics_data['z'].append(-1)
                antibiotics_data['x'].append(x)
                antibiotics_data['h'].append(history)

        self.n_a = self.get_n_a()
        print("{} different antibiotics".format(self.n_a))
        self.n_training_samples = len(patients)
        print("{} patients".format(self.n_training_samples))
        self.antibiotics_training_data = antibiotics_data
        print("Organisms: {}".format(self.organism_to_x_dict.keys()))
        return antibiotics_data

    def antibiotic_to_treatment(self, antibiotic):
        if antibiotic in self.antibiotic_to_treatment_dict:
            index = self.antibiotic_to_treatment_dict[antibiotic]
        else:
            index = self.antibiotic_counter
            self.antibiotic_to_treatment_dict[antibiotic] = index
            self.antibiotic_counter += 1
        return index

    def too_many_x(self, organism):
        if organism in self.organism_to_x_dict:
            return False
        else:
            if self.x_counter < 2**self.n_x:
                conversion = '{}0:0{}b{}'.format('{', str(self.n_x), '}')
                self.organism_to_x_dict[organism] = np.array([int(s) for s in list((conversion.format(self.x_counter)))])  # Convert to list of binary
                self.x_counter += 1
                return False
            else:
                return True

    def get_test_data(self, nr_test_samples=0):
        xs = self.antibiotics_training_data['x']
        histories = self.antibiotics_training_data['h']
        xs, histories = self.shuffle_histories(xs, histories)
        xs = xs[:nr_test_samples]
        histories = histories[:nr_test_samples]

        data = []
        for i, history in enumerate(histories):
            z = -1
            x = xs[i]
            subject = [z, x, np.ones(self.n_a)*-1]

            for intervention in history:
                treatment, outcome = intervention
                subject[2][treatment] = outcome
            data.append(np.array(subject))
        self.antibiotics_test_data = np.array(data)
        return self.antibiotics_test_data

    def shuffle_histories(self, xs, histories):
        patients = list(zip(xs, histories))
        self.random.shuffle(patients)
        xs, histories = zip(*patients)
        return xs, histories


def interpretation_to_outcome(interpretation):


    if interpretation == 'S':
        return 2
    elif interpretation == 'I':
        return 1
    elif interpretation == 'R':
        return 0
    elif interpretation == 'P':
        return None


