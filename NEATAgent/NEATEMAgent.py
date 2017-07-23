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
        self.alpha = 0.1
        self.beta = 0.01

    def get_value_function(self):
        return self.valueFunction

    def get_policy(self):
        return self.policy

    def get_network(self):
        return self.neural_net

    def get_fitness(self):
        return self.fitness

    def update_value_function(self, old_state, new_state, reward):
        # calculate TD error. For critic
        old_state_features = self.neural_net.activate(old_state)
        new_state_features = self.neural_net.activate(new_state)
        delta = reward \
            + self.gamma * self.valueFunction.get_value(new_state_features) \
            - self.valueFunction.get_value(old_state_features)

        # Update critic parameters
        delta_omega = np.dot((self.alpha * delta), np.array(old_state_features))
        self.valueFunction.update_parameters(delta_omega)

    def update_policy_function(self, state_transitions):
        '''
        Need to update the policy parameters which was used to make the action.
        There are N number of discrete actions.
        This N number is stored in Policy so I should be looking to put this method there 
        :param state_transitions: 
        :return:
        '''
        # Update the policy parameters for the actions that are taken

        # Create a delta vector of size # actions where each element is a tuple (component1 and component 2)
        delta = np.array([DeltaPolicy() for _ in range(self.policy.get_num_actions())])

        for j, state_transition in enumerate(state_transitions):
            phi = self.neural_net.activate(state_transition.get_start_state())
            _, actions_distribution = self.policy.get_action(phi)

            action_prob = actions_distribution[state_transition.get_action()]

            component1 = np.dot(np.dot(action_prob * (action_prob - 1), phi), np.transpose(phi))

            phi_new = self.neural_net.activate(state_transition.get_end_state())
            td_error = state_transition.get_reward() + self.gamma * self.valueFunction.get_value(phi_new) - self.valueFunction.get_value(phi)

            component2 = np.dot((1 - action_prob) * td_error, phi)

            delta[state_transition.get_action()].add(component1, component2)

        for i in range(len(delta)):
            delta[i].component1 = delta[i].component1 / delta[i].state_transition_count
            delta[i].component2 = np.dot(2 / delta[i].state_transition_count, delta[i].component2)
            delta[i].calculate_delta()

        self.policy.update_parameters(delta) # delta is a vector of size (num of actions) and each element is a vector of policy parameter


class DeltaPolicy(object):

    def __init__(self):
        self.component1 = 0
        self.component2 = 0
        self.delta = 0
        self.state_transition_count = 0

    def add(self, component1, component2):
        self.component1 += component1
        self.component2 += component2
        self.state_transition_count += 1

    def calculate_delta(self):
        self.delta = np.dot(self.component1, self.component2)

