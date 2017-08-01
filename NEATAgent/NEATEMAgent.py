from ValueFunction.ValueFunction import ValueFunction
from Policy.SoftmaxPolicy import SoftmaxPolicy
import numpy as np


class NeatEMAgent(object):
    def __init__(self, neural_net, dimension, num_actions):
        self.neural_net = neural_net
        self.dimension = dimension
        self.valueFunction = ValueFunction(dimension)
        self.policy = SoftmaxPolicy(dimension, num_actions)
        self.fitness = 0
        self.gamma = 0.99

    def get_value_function(self):
        return self.valueFunction

    def get_policy(self):
        return self.policy

    def get_network(self):
        return self.neural_net

    def get_fitness(self):
        return self.fitness

    def update_value_function(self, state_transitions):
        delta_omega = np.zeros(self.dimension, dtype=float)

        for state_transition in state_transitions:
            old_state_features = self.neural_net.activate(state_transition.get_start_state())
            new_state_features = self.neural_net.activate(state_transition.get_end_state())
            derivative = 2 * (self.valueFunction.get_value(old_state_features) - (state_transition.get_reward() + self.gamma * self.valueFunction.get_value(new_state_features)))
            delta = np.dot(derivative, self.valueFunction.get_dimension())
            delta_omega += delta

        delta_omega /= len(state_transition)
        self.valueFunction.update_parameters(delta_omega)

    def update_policy_function(self, trajectories):
        '''
        Need to update the policy parameters which was used to make the action.
        There are N number of discrete actions.
        This N number is stored in Policy so I should be looking to put this method there 
        :param trajectories: 
        :return:
        '''
        # Update the policy parameters for the actions that are taken

        # Create a delta vector of size [# actions] where each element is a delta policy object
        delta = np.array([DeltaPolicy(self.dimension) for _ in range(self.policy.get_num_actions())])

        for j, (total_reward, _, state_transitions) in enumerate(trajectories):
            for state_transition in state_transitions:
                phi = self.neural_net.activate(state_transition.get_start_state())
                _, actions_distribution = self.policy.get_action(phi)

                action_prob = actions_distribution[state_transition.get_action()]
                # check for matrix multiplication
                phi_dot = np.dot(action_prob * (action_prob - 1), np.array(phi))
                # convert vector into a column vector
                phi_dot = phi_dot.reshape(-1, 1)

                component1 = np.matmul(phi_dot, phi_dot.transpose())  # produces a dimension * dimension matrix

                phi_new = self.neural_net.activate(state_transition.get_end_state())
                td_error = state_transition.get_reward() + self.gamma * self.valueFunction.get_value(phi_new) - self.valueFunction.get_value(phi)

                component2 = np.dot((1 - action_prob) * td_error, phi)

                # we only update the policy parameter that was used to perform the action
                delta[state_transition.get_action()].add(component1, component2)

        for i in range(len(delta)):
            delta[i].component1 = delta[i].component1 / len(trajectories) # delta[i].state_transition_count
            delta[i].component2 = np.dot(2.0 / len(trajectories), delta[i].component2)
            delta[i].calculate_delta()

        self.policy.update_parameters(delta)  # delta is a vector of size (num of actions) and each element is a vector of policy parameter


class DeltaPolicy(object):

    def __init__(self, dimension):
        self.component1 = 0
        self.component2 = np.zeros(dimension, dtype=float)
        self.delta = np.zeros(dimension, dtype=float)
        self.state_transition_count = 0.0

    def add(self, component1, component2):
        self.component1 += component1
        self.component2 += component2
        self.state_transition_count += 1.0

    def calculate_delta(self):
        self.delta = np.dot(self.component1, self.component2)

