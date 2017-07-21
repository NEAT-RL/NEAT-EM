import random
import numpy as np
import math


class SoftmaxPolicy(object):
    def __init__(self, dimension):
        self.dimension = dimension
        self.parameters = []
        self.parameters = np.zeros((3, self.dimension), dtype=float)
        self.sigma = 1.0

    def get_action(self, state_feature):
        '''
        Perform dot product between state feature and policy parameter and return sample from the normal distribution
        :param state_feature: 
        :return: 
        '''

        # for each policy parameter (representing each action)
        # calculate phi /cdot theta
        # put these into array and softmax and compute random sample
        action_probabilities = []
        for i, parameter in enumerate(self.parameters):
            mu = np.dot(state_feature, parameter)
            action_probabilities.append(mu)

        # substract the largest value of actions to avoid erroring out when trying to find exp(value)
        max_value = action_probabilities[np.argmax(action_probabilities)]
        for i in range(len(action_probabilities)):
            action_probabilities[i] = action_probabilities[i] - max_value

        softmax = np.exp(action_probabilities) / np.sum(np.exp(action_probabilities), axis=0)

        # I could pre process
        p = random.uniform(0, 1)
        cumulative_probability = 0.0
        chosen_policy_index = 0
        for n, prob in enumerate(softmax):
            cumulative_probability += prob
            if p <= cumulative_probability:
                chosen_policy_index = n
                break

        return chosen_policy_index, softmax

    def update_parameters(self, delta):
        for i, parameter in enumerate(self.parameters):
            for j, param in enumerate(parameter):
                parameter[j] = param + delta[j]

    def dlogPi(self, state_features, action):
        mu = np.dot(state_features, self.parameters)
        deratives = np.zeros(len(state_features))

        component1 = (action - mu) / math.pow(self.sigma, 2)

        for i, state_feature in enumerate(state_features):
            deratives[i] = state_feature * component1

        # np.dot(component1, state_feature)

        return deratives
