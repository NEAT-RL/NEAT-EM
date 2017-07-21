from ValueFunction.ValueFunction import ValueFunction
from Policy.SoftmaxPolicy import SoftmaxPolicy
import numpy as np


class NeatEMAgent(object):
    def __init__(self, neural_net, dimension):
        self.neural_net = neural_net
        self.dimension = dimension
        self.valueFunction = ValueFunction(dimension)
        self.policy = SoftmaxPolicy(dimension)
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
        delta = reward + self.gamma * self.valueFunction.get_value(new_state_features) - self.valueFunction.get_value(
            old_state_features)

        # Update critic parameters
        delta_omega = np.dot((self.alpha * delta), np.array(old_state_features))
        self.valueFunction.update_parameters(delta_omega)

    def update_policy_function(self, trajectories):
        component1 = 0
        component2 = 0
        for i, (total_reward, _, state_trasitions) in enumerate(trajectories):
            for j, (state, action, reward, new_state) in enumerate(state_trasitions):
                phi = self.neural_net.activate(state)
                pi, actions_distribution = self.policy.get_action(phi)
                action = actions_distribution[pi]
                component1 += np.dot(np.dot(action * (action - 1), phi), np.transpose(phi))

                phi_new = self.neural_net.activate(new_state)
                delta = reward + self.gamma * self.valueFunction.get_value(phi_new) - self.valueFunction.get_value(phi)

                component2 += np.dot((1-action) * delta, phi)

        component1 = component1 / len(trajectories)
        component2 = np.dot(2 / len(trajectories), component2)

        # for i, trajectory in enumerate(trajectories):
        #     for i, state_transition in trajectory:
        policy_parameter_update = np.dot(component1, component2)
        self.policy.update_parameters(policy_parameter_update)