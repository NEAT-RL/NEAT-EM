import random
import numpy as np
import math
import logging
from datetime import datetime
import scipy.stats as stats

logging.basicConfig(filename='log/policy-debug-' + str(datetime.now()) + '.log', level=logging.DEBUG)
logger = logging.getLogger()


class SoftmaxPolicy(object):
    def __init__(self, dimension, num_actions):
        self.dimension = dimension
        self.parameters = []
        self.num_actions = num_actions
        self.sigma = 1.0
        self.default_learning_rate = 0.001
        self.kl_threshold = 0.001
        self.tiny = 1e-8
        self.initialise_parameters()

    def initialise_parameters(self):
        """
        TODO: See different ways of initialising the parameters.
         - Zero vectors
         - Random vectors (capped to [-10, 10] for example)
         - Maximising log likelihood etc
        :return: 
        """
        self.parameters = np.random.uniform(low=self.tiny, high=1, size=(self.num_actions, self.dimension))
        # self.parameters = np.zeros(shape=(self.num_actions, self.dimension), dtype=float)
        # self.parameters.fill(self.tiny)

    def get_num_actions(self):
        return self.num_actions

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
        """
        Delta is an array where each element is delta for a policy parameter.
        Note: Number of policy parameters = number of actions.
        Each delta object contains a delta of the policy parameter.
        :param delta: 
        :return:
        Assume size of delta == number of actions
        """
        # Calculate KL-divergence

        new_parameters = np.zeros(shape=(self.num_actions, self.dimension), dtype=float)

        for i in range(len(self.parameters)):
            new_parameters[i] = self.__calculate_gradient(self.parameters[i], delta[i])

        for i in range(len(self.parameters)):
            learning_rate = self.default_learning_rate
            for j in range(10):
                kl_difference = stats.entropy(new_parameters[i], self.parameters[i])
                if kl_difference < self.kl_threshold:
                    self.parameters[i] = new_parameters[i]
                    break
                else:
                    # logger.debug("Not updating policy parameter as kl_difference was %f. Learning rate=%f",
                    #              kl_difference, learning_rate)
                    learning_rate /= 10  # reduce learning rate
                    # recalculate gradient using the new learning rate
                    new_parameters[i] = self.__calculate_gradient(self.parameters[i], delta[i], learning_rate)

    def __calculate_gradient(self, parameter, delta_vector, learning_rate=None):
        new_parameter = np.zeros(shape=self.dimension, dtype=float)

        if learning_rate is None:
            learning_rate = self.default_learning_rate

        for j, param in enumerate(parameter):
            new_value = max(min(param - learning_rate * delta_vector.delta[j], 10), -10)
            new_parameter[j] = new_value + self.tiny  # adding tiny here to avoid getting potential 0 value

        return new_parameter
