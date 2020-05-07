import psycopg2
import numpy as np
import base64


class AntibioticsDatabase:
    def __init__(self, n_x=2, antibiotic_limit=8, seed=None):
        self.random = np.random.RandomState()
        self.random.seed(seed)
        self.antibiotic_to_treatment_dict = {}
        self.antibiotic_counter = 0
        self.n_x = n_x
        self.antibiotic_limit = antibiotic_limit
        self.x_counter = 0
        self.organism_to_x_dict = {}
        self.n_training_samples = None
        self.antibiotics_training_data = None
        self.antibiotics_test_data = None
        self.name = 'Antibiotics'
        self.n_a = None
        self.n_y = 3
        self.max_outcome = self.n_y-1

        pw = base64.b64decode(b'aGVhbHRoeUFJ').decode("utf-8")
        self.conn = psycopg2.connect(database="mimic", user="postgres", password=pw)
        self.cur = self.conn.cursor()
        self.cur.execute('set search_path=mimiciii')

        self.treatment_to_test = {
            'Caspofungin': None,
            'Cefazolin': 'CEFAZOLIN',
            'Cefepime': 'CEFEPIME',
            'Ceftazidime': 'CEFTAZIDIME',
            'Ceftriaxone': 'CEFTRIAXONE',
            'Chloroquine': None,
            'Ciprofloxacin': 'CIPROFLOXACIN',
            'Clindamycin': 'CLINDAMYCIN',
            'Colistin': None,
            'Daptomycin': 'DAPTOMYCIN',
            'Doxycycline': 'TETRACYCLINE',
            'Erythromycin': 'ERYTHROMYCIN',
            'Ethambutol': None,
            'Fluconazole': None,
            'Foscarnet': None,
            'Gancyclovir': None,
            'Gentamicin': 'GENTAMICIN',
            'Imipenem/Cilastatin': 'IMIPENEM',
            'Isoniazid': None,
            'Keflex': 'CEFAZOLIN',  # Maybe check
            'Levofloxacin': 'LEVOFLOXACIN',
            'Linezolid': 'LINEZOLID',
            'Mefloquine': None,
            'Meropenem': 'MEROPENEM',
            'Metronidazole': None,
            'Micafungin': None,
            'Moxifloxacin': 'CIPROFLOXACIN',  # Check!
            'Nafcillin': 'PENICILLIN',  # Check!
            'Oxacillin': 'OXACILLIN',
            'Penicillin G potassium': 'PENICILLIN G',
            'Piperacillin': 'PIPERACILLIN',
            'Piperacillin/Tazobactam (Zosyn)': 'PIPERACILLIN/TAZO',
            'Pyrazinamide': None
            'Quinine': None,


        }
        self.temp()

    def get_n_a(self):
        return min(self.antibiotic_counter, self.antibiotic_limit)

    def get_data(self):
        self.cur.execute('SELECT hadm_id, org_name, ab_name, interpretation FROM microbiologyevents WHERE ab_name IS NOT NULL;')
        microbiology_test_data = self.cur.fetchall()
        self.cur.execute("SELECT DISTINCT(label), hadm_id FROM inputevents_mv JOIN d_items ON inputevents_mv.itemid = d_items.itemid WHERE d_items.category like 'Antibiotics'")
        used_antibiotics = self.cur.fetchall()
        print(used_antibiotics)
        patients = {}
        self.random.shuffle(microbiology_test_data)
        for chartevent in microbiology_test_data:
            hadm_id = chartevent[0]
            organism = chartevent[1]
            treatment_name = str.lower(chartevent[2])
            treatment = self.antibiotic_to_treatment(treatment_name)
            outcome = interpretation_to_outcome(chartevent[3])
            if not self.too_many_x(organism) and treatment < self.antibiotic_limit and outcome is not None:
                intervention = np.array([treatment, outcome])
                try:
                    if treatment not in [intervention[0] for intervention in patients[hadm_id][organism]]:
                        patients[hadm_id][organism].append(intervention)

                except KeyError:
                    patients[hadm_id] = {organism: [intervention]}
        input_patients = {}
        for chartevent in used_antibiotics:
            treatment_name = str.lower(chartevent[0])
            treatment = self.antibiotic_to_treatment(treatment_name)
            hadm_id = chartevent[1]
            if hadm_id in patients:
                if len(patients[hadm_id].keys()):
                    pass
                    # print('Warning: more than one bacteria')
                for organism in patients[hadm_id].keys():
                    if hadm_id in input_patients:
                        if hadm_id in input_patients:
                            input_patients[hadm_id][organism].append(treatment)
                        else:
                            input_patients[hadm_id][organism] = [treatment]
                    else:
                        input_patients[hadm_id] = {organism: [treatment]}

        #print(input_patients)

        ## clean data

        self.n_a = self.get_n_a()

        antibiotics_data = {'z': [], 'x': [], 'h': []}
        test_data = []
        for hadm_id, microbiology_test_data in patients.items():
            for organism, history in microbiology_test_data.items():
                x = self.organism_to_x_dict[organism]
                test_data.append(self.get_test_data(x, history))

        for hadm_id, used_antib in input_patients.items():
            for organism, history in used_antib.items():
                x = self.organism_to_x_dict[organism]
                antibiotics_data['z'].append(-1)
                antibiotics_data['x'].append(x)
                antibiotics_data['h'].append(history)

        print("{} different antibiotics".format(self.n_a))
        self.n_training_samples = len(patients)
        print("{} patients".format(self.n_training_samples))
        self.antibiotics_training_data = antibiotics_data
        print("{} patients in training data, {} in test data".format(len(antibiotics_data['x']), len(test_data)))
        print("{} organisms".format(self.x_counter))
        print("Organisms: {}".format(self.organism_to_x_dict.keys()))
        return antibiotics_data, test_data

    def temp(self):
        self.cur.execute("SELECT label from d_items WHERE LOWER(category) like '%anti%' ORDER BY category ,label;")
        data = self.cur.fetchall()
        print(type(data))
        for treatment in data:
            print(treatment)

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

    def get_test_data(self, x, history):
        z = -1
        subject = [z, x, np.ones(self.n_a)*-1]

        for intervention in history:
            treatment, outcome = intervention
            subject[2][treatment] = outcome
        return subject

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



