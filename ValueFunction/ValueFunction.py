import numpy as np


class ValueFunction(object):
    def __init__(self, dimension):
        self.dimension = dimension
        self.parameters = np.zeros(dimension, dtype=float)
        # self.parameters.fill(1e-8)  # do not use zero vector
        self.beta = 0.1

    def get_value(self, state_feature):
        return np.dot(self.parameters, state_feature)

    def get_parameter(self):
        return np.copy(self.parameters)

    def update_parameters(self, delta):
        self.parameters = self.parameters + self.beta * delta
