import numpy as np


class ValueFunction(object):
    def __init__(self, dimension):
        self.dimension = dimension
        self.parameters = np.zeros(dimension, dtype=float)
        self.beta = 0.01

    def get_value(self, state_feature):
        return np.dot(self.parameters, state_feature)

    def update_parameters(self, delta):
        for i, param in enumerate(self.parameters):
            self.parameters[i] = param + self.beta * delta[i]
