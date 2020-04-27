

class DistAlgWrapper:
    def __init__(self, dist, name="Distribution", label="dist"):
        self.dist = dist
        self.name = name
        self.label = label

    def learn(self):
        pass

    def evaluate(self, patient):
        return self.dist.evaluate(patient)