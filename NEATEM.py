from __future__ import print_function

import multiprocessing
import os
import pickle
import argparse
import logging
import random
import sys
import neat
import gym
import gym.wrappers as wrappers
import configparser
import visualize
from NEATEMAgent import NeatEMAgent
import numpy as np
import heapq
from datetime import datetime


class StateTransition(object):
    def __init__(self, start_state, action, reward, end_state):
        self.start_state = start_state
        self.action = action
        self.reward = reward
        self.end_state = end_state

    def __hash__(self):
        return hash((self.start_state, self.end_state))

    def __eq__(self, other):
        return self.start_state == other.state and self.end_state == other.new_state

    def get_start_state(self):
        return self.start_state

    def get_end_state(self):
        return self.end_state

    def get_action(self):
        return self.action

    def get_reward(self):
        return self.reward

    def get_tuple(self):
        return (self.start_state)


class NeatEM(object):
    def __init__(self, config):
        pop = neat.Population(config)
        self.stats = neat.StatisticsReporter()
        pop.add_reporter(self.stats)
        pop.add_reporter(neat.StdOutReporter(True))
        # Checkpoint every 10 generations or 900 seconds.
        pop.add_reporter(neat.Checkpointer(10, 900))
        self.config = config
        self.population = pop
        self.pool = multiprocessing.Pool()
        self.trajectories = []
        self.state_transitions = set()
        self.initialise_trajectories(props.getint('initialisation', 'trajectory_size'))

    def initialise_trajectories(self, num_trajectories):
        '''
        Initialise trajectories of size: 'size'.
        Each trajectory we store N number of state transitions. (state, reward, next state, action)
        :param num_trajectories: 
        :return: 
        '''
        max_steps = props.getint('initialisation', 'max_steps')

        for i in range(num_trajectories):
            trajectory = []
            state = env.reset()
            terminal_reached = False
            steps = 0
            reward_count = 0
            while not terminal_reached and steps < max_steps:
                # sample action from the environment
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)
                state_transition = StateTransition(state, action, reward, next_state)
                # insert state transition to the trajectory
                self.state_transitions.add(state_transition)
                trajectory.append(state_transition)
                reward_count += reward
                state = next_state
                if done:
                    terminal_reached = True
                steps += 1

            # we have to insert timestamp as second entry so that we can order trajectories with the same reward count
            heapq.heappush(self.trajectories, (-reward_count, datetime.now(), trajectory))

    def execute_algorithm(self, generations):
        self.population.run(self.fitness_function, generations)

    def fitness_function(self, genomes, config):
        '''
        This method is called every generation.
        Create new array 
        :param genomes: 
        :param config: 
        :return: 
        '''

        nets = []
        for genome_id, genome in genomes:
            # reinitialise the agents
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            neatNetwork = NeatEMAgent(network, props.getint('neuralnet', 'dimension'))

            nets.append((genome, neatNetwork))
            genome.fitness = 0


        # select K random agents to perform rollout

        num_new_trajectories = props.getint('evaluation', 'new_trajectories')
        max_steps = props.getint('initialisation', 'max_steps')
        rand_policy = random.randint(0, len(nets)-1)
        for i in range(num_new_trajectories):
            genome, agent = nets[rand_policy]
            # perform a rollout
            state = env.reset()
            terminal_reached = False
            steps = 0
            reward_count = 0
            new_trajectory = []
            while not terminal_reached and steps < max_steps:
                # env.render()
                state_features = agent.get_network().activate(state)
                action, actions_distribution = agent.get_policy().get_action(state_features)
                next_state, reward, done, info = env.step(action)
                # insert state transition to the trajectory
                state_transition = StateTransition(state, action, reward, next_state)
                new_trajectory.append(state_transition)
                reward_count += reward
                state = next_state
                if done:
                    terminal_reached = True
                steps += 1

            heapq.heappush(self.trajectories, (-reward_count, datetime.now(), new_trajectory))

            new_policy = random.randint(0, len(nets) - 1)
            while rand_policy == new_policy:
                new_policy = random.randint(0, len(self.trajectories) - 1)
            rand_policy = new_policy

        # strip weak trajectories from trajectory_set and add state transitions to set state_transitions
        self.trajectories = self.trajectories[:10]
        for i in range(len(self.trajectories)):
            _, __, trajectory = self.trajectories[i]
            self.state_transitions = self.state_transitions | set(trajectory)

        # For each individual in the population
        for genome, net in nets:
            random_trajectory = random.randint(0, len(self.trajectories) - 1)
            # 20 = number of state transitions we use for experience replay
            for i in range(20):
                _, _, state_transition = self.trajectories[random_trajectory]
                random_state_transition = random.randint(0, len(state_transition) - 1)
                state, action, reward, next_state = state_transition[random_state_transition]
                # update TD error and value function
                net.update_value_function(state, next_state, reward)

                new_random_trajectory = random.randint(0, len(self.trajectories) - 1)
                while new_trajectory == new_random_trajectory:
                    new_random_trajectory = random.randint(0, len(self.trajectories) - 1)


            # update policy parameter
            # we only update the policy parameter if it was used for the action
            net.update_policy_function(self.trajectories)

            # now assign fitness to each individual/genome
            # fitness is the log prob of following the best trajectory
            # I need the get action to return me the probabilities of the actions rather than a numerical action
            best_trajectory = self.trajectories[0]
            best_trajectory_prob = 1
            total_reward, _, state_transitions = best_trajectory
            for j, (state, action, reward, new_state) in enumerate(state_transitions):
                # calculcate probability of the action probability where policy action = action
                state_features = agent.get_network().activate(state)
                _, actions_distribution = agent.get_policy().get_action(state_features)
                best_trajectory_prob *= actions_distribution[action]

            genome.fitness = np.log(best_trajectory_prob)


def save_best_genomes(best_genomes, has_won):

    for n, g in enumerate(best_genomes):
        name = "results/"
        if has_won:
            name += 'winner-{0}'.format(n)
        else:
            name += 'best-{0}'.format(n)

        with open(name + '.pickle', 'wb') as f:
            pickle.dump(g, f)

        visualize.draw_net(config, g, view=False, filename=name + "-net.gv")
        visualize.draw_net(config, g, view=False, filename=name + "-net-enabled.gv",
                           show_disabled=False)
        visualize.draw_net(config, g, view=False, filename=name + "-net-enabled-pruned.gv",
                           show_disabled=False, prune_unused=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MountainCar-v0', help='Select the environment to run')
    args = parser.parse_args()

    gym.undo_logger_setup()
    logging.basicConfig(filename='debug.log', level=logging.DEBUG)
    logger = logging.getLogger()
    logging.Formatter('[%(asctime)s] %(message)s')
    env = gym.make(args.env_id)

    logger.debug("action space: %s", env.action_space)
    logger.debug("observation space: %s", env.observation_space)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/neat-em-data/' + str(datetime.now())
    env = wrappers.Monitor(env, directory=outdir, force=True)

    # load properties
    props = configparser.ConfigParser()
    props.read('neatem_properties.ini')

    # run the algorithm

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    population = NeatEM(config)

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    while 1:
        try:
            # Run for 5 generations
            population.execute_algorithm(props.getint('neuralnet', 'generation'))

            visualize.plot_stats(population.stats)
            # Use the five best genomes seen so far as an ensemble-ish control system.
            best_genomes = population.stats.best_unique_genomes(5)
            best_networks = []

            save_best_genomes(best_genomes, True)
            break

        except KeyboardInterrupt:
            logger.debug("User break.")
            # save the best neural network or save top 5?
            best_genomes = population.stats.best_unique_genomes(5)

            save_best_genomes(best_genomes, False)
            break

    env.close()

    # Upload to the scoreboard. We could also do this from another
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
