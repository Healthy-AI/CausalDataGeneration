

class Doctor:
    def __init__(self, data):
        self.name = 'Doctor'
        self.label = 'Doctor'
        self.evaluations = list(data['h'])
        self.patientindex = 0


    def learn(self):
        self.patientindex = 0

    def evaluate(self, patient):
        eval = self.evaluations[self.patientindex]
        self.patientindex += 1
        return eval