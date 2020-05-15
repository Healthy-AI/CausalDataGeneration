

class Doctor:
    def __init__(self, data=None):
        self.name = 'Doctor'
        self.label = 'Doctor'
        if data is not None:
            self.evaluations = list(data['h'])
        self.patientindex = 0

    def learn(self):
        self.patientindex = 0

    def evaluate(self, patient):
        eval = self.evaluations[self.patientindex]
        self.patientindex += 1
        return eval

    def set_data(self, data):
        self.evaluations = list(data['h'])
