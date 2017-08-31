from ValueFunction.ValueFunction import ValueFunction
from Policy.SoftmaxPolicy import SoftmaxPolicy
import numpy as np
from datetime import datetime
from . import Feature
import math
"""
Mountain car
state = (position, velocity)
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        min speed = -self.max_speed = -0.07
         
TODO: 
Add Cartpole settings
"""


class NeatEMAgent(object):
    def __init__(self, network, dimension, num_actions):
        self.num_actions = num_actions
        self.dimension = dimension
        # Neat
        self.feature = Feature.NEATFeature(network)
        # Mountain Car
        # self.feature = self.__create_discretised_feature([5, 5], 2, [-1.2, -0.07], [0.6, 0.07])
        # CartPole-v0
        # self.feature = self.__create_discretised_feature([10, 10, 10, 10], 4, [-2.4, -10, -41.8 * math.pi / 180, -10], [2.4, 10, 41.8 * math.pi / 180, 10])
        self.valueFunction = ValueFunction(dimension)
        self.policy = SoftmaxPolicy(dimension, num_actions, self.feature)
        self.fitness = 0
        self.gamma = 0.99
        self.max_total_reward = 0.0
        self.best_policy_parameters = np.zeros(shape=(self.dimension * self.num_actions))

    @staticmethod
    def __create_discretised_feature(partition_size, state_length, state_lower_bounds, state_upper_bounds):
        """
        :param partition_size: Array of partition_sizes for each state field
        :param state_length: Number of states == input dimension (state)
        :param state_lower_bounds: array of lower bounds for each state field
        :param state_upper_bounds: array of upper bounds for each state field
        :return: discretised feature
        """
        intervals = []
        output_dimension = 0

        for i in range(state_length):
            output_dimension += partition_size[i]
            state_col = Feature.DiscretizedFeature.create_partition(state_lower_bounds[i],
                                                                    state_upper_bounds[i], partition_size[i])
            intervals.append(state_col)
        return Feature.DiscretizedFeature(state_length, output_dimension, intervals)

    def get_feature(self):
        return self.feature

    def get_value_function(self):
        return self.valueFunction

    def get_policy(self):
        return self.policy

    def get_fitness(self):
        return self.fitness

    def update_value_function(self, state_transitions):
        delta_omega = np.zeros(self.dimension, dtype=float)

        for state_transition in state_transitions:
            old_state_features = self.feature.phi(state_transition.get_start_state())
            new_state_features = self.feature.phi(state_transition.get_end_state())
            derivative = 2 * (self.valueFunction.get_value(old_state_features) - (state_transition.get_reward() + self.gamma * self.valueFunction.get_value(new_state_features)))
            delta = np.dot(derivative, self.valueFunction.get_parameter())
            delta_omega += delta

        delta_omega /= len(state_transitions)
        self.valueFunction.update_parameters(delta_omega)

    def update_policy_function(self, random_state_transitions, all_state_transitions):
        """
        For each parameter in error squared function:
           e(theta + delta)Transpose cdot e(theta+delta) - e(theta-delta)/(2*delta)
        """
        t_start = datetime.now()
        # first copy the policy parameters
        original_policy_parameters = self.policy.get_policy_parameters()

        d_error_squared = [self.approximate_d_error_squared(i, random_state_transitions, original_policy_parameters) for i in
                           range(len(original_policy_parameters))]

        # set policy parameter to its original value in case it has been changed. WHICH IT SHOULDN'T BE.
        self.policy.set_policy_parameters(original_policy_parameters)
        # update policy parameter
        self.policy.update_parameters(d_error_squared, all_state_transitions)
        print("{0}. Updated policy parameter. Time taken: {1}".format(datetime.now(),
                                                                      (datetime.now() - t_start).total_seconds()))

    def approximate_d_error_squared(self, index, random_state_transitions, original_policy_parameters):
        delta = 0.01
        # make new policy with original_policy_parameters
        policy = SoftmaxPolicy(self.dimension, self.num_actions, self.feature, self.policy.is_greedy)  # feature is unused if KL divergence is unused
        policy.set_policy_parameters(original_policy_parameters)

        # maintain positive delta and negative delta error functions
        error_func_positive_delta = np.zeros(shape=(policy.num_actions * self.dimension), dtype=float)
        error_func_negative_delta = np.zeros(shape=(policy.num_actions * self.dimension), dtype=float)

        new_parameters_positive_delta = np.copy(original_policy_parameters)
        new_parameters_negative_delta = np.copy(original_policy_parameters)

        # add/subtract delta to both positive delta and negative delta function parameters
        new_parameters_positive_delta[index] = new_parameters_positive_delta[index] + delta
        new_parameters_negative_delta[index] = new_parameters_negative_delta[index] - delta

        # calculate the error function
        for state_transition in random_state_transitions:
            phi_start = self.feature.phi(state_transition.get_start_state())

            # set theta + delta and calculate dlogpi for positive delta case
            policy.set_policy_parameters(new_parameters_positive_delta)
            dlogpi_positive_delta = policy.dlogpi(phi_start, state_transition.get_action())

            # set theta - delta and calculate dlogpi for negative delta case
            policy.set_policy_parameters(new_parameters_negative_delta)
            dlogpi_negative_delta = policy.dlogpi(phi_start, state_transition.get_action())

            # calculate td_error. TD Error calculate is independent of policy
            phi_end = self.feature.phi(state_transition.get_end_state())
            td_error = state_transition.get_reward() + self.gamma * self.valueFunction.get_value(
                phi_end) - self.valueFunction.get_value(phi_start)

            # Multiply dlogpi with td error for positive delta and negative delta functions
            dlogpi_positive_delta *= td_error
            dlogpi_negative_delta *= td_error

            # added to error function positive and error function negative
            error_func_positive_delta += dlogpi_positive_delta
            error_func_negative_delta += dlogpi_negative_delta

        error_func_positive_delta /= len(random_state_transitions)
        error_func_negative_delta /= len(random_state_transitions)

        '''
        now calculate scalar approximation.
        e(theta + delta) <==> error_func_negative_delta
        e(theta - delta) <==> error_func_negative_delta

        e(theta + delta)^Transpose dot e(theta+delta) - e(theta-delta)^Transpose dot e(theta-delta)/(2*delta)
        '''
        error_derivative = np.dot(np.transpose(error_func_positive_delta), error_func_positive_delta) - np.dot(
            np.transpose(error_func_negative_delta), error_func_negative_delta)
        error_derivative /= (2 * delta)

        return error_derivative
