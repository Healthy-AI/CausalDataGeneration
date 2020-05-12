

class Doctor:
    def __init__(self, evaluations):
        self.name = 'Doctor'
        self.label = 'Doc'
        self.evaluations = evaluations
        self.patientindex = 0

    def learn(self):
        pass

    def evaluate(self, patient):
        eval = self.evaluations[self.patientindex]
        self.patientindex += 1
        return eval