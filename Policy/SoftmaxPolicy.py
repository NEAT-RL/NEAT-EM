import random
import numpy as np
import math
import logging
from datetime import datetime

logging.basicConfig(filename='log/policy-debug-' + str(datetime.now()) + '.log', level=logging.DEBUG)
logger = logging.getLogger()

class SoftmaxPolicy(object):
    def __init__(self, dimension, num_actions):
        self.dimension = dimension
        self.parameters = []
        self.num_actions = num_actions
        self.sigma = 1.0
        self.initialise_parameters()

    def initialise_parameters(self):
        """
        TODO: See different ways of initialising the parameters.
         - Zero vectors
         - Random vectors (capped to [-10, 10] for example)
         - Maximising log likelihood etc
        :return: 
        """
        self.parameters = np.zeros((self.num_actions, self.dimension), dtype=float)

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
        for i, parameter in enumerate(self.parameters):
            capped_value = False
            for j, param in enumerate(parameter):
                new_value = max(min(param + delta[i].delta[j], 10), -10)
                if math.fabs(new_value) == 10:
                    logger.debug("Capped parameter value from %f, to: %d", param + delta[i].delta[j], new_value)
                    capped_value = True
                parameter[j] = max(min(param + delta[i].delta[j], 10), -10)

            if capped_value:
                logger.debug("Policy Parameter was capped: %s", parameter)
