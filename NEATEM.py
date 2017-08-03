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
from NEATAgent.NEATEMAgent import NeatEMAgent
import numpy as np
import heapq
from datetime import datetime
import csv

class StateTransition(object):
    def __init__(self, start_state, action, reward, end_state):
        self.start_state = start_state
        self.action = action
        self.reward = reward
        self.end_state = end_state

    def __hash__(self):
        return hash(str(np.concatenate((self.start_state, self.end_state))))

    def __eq__(self, other):
        return np.array_equal(self.start_state, other.start_state) and np.array_equal(self.end_state, other.end_state)

    def get_start_state(self):
        return self.start_state

    def get_end_state(self):
        return self.end_state

    def get_action(self):
        return self.action

    def get_reward(self):
        return self.reward

    def get_tuple(self):
        return self.start_state, self.action, self.reward, self.end_state


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
        self.initialise_trajectories(props.getint('initialisation', 'trajectory_size'))
        self.generation_count = 0
        self.best_agents = []

    def initialise_trajectories(self, num_trajectories):
        '''
        Initialise trajectories of size: 'size'.
        Each trajectory we store N number of state transitions. (state, reward, next state, action)
        :param num_trajectories:
        :return:
        '''
        logger.debug("Initialising trajectories")
        tstart = datetime.now()

        max_steps = props.getint('initialisation', 'max_steps')
        step_size = props.getint('initialisation', 'step_size')
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

                for x in range(step_size - 1):
                    if done:
                        terminal_reached = True
                        break
                    next_state, reward2, done, info = env.step(action)
                    reward += reward2

                state_transition = StateTransition(state, action, reward, next_state)
                # insert state transition to the trajectory
                trajectory.append(state_transition)
                reward_count += reward
                state = next_state
                steps += 1
                if done:
                    terminal_reached = True

            # we have to insert timestamp as second entry so that we can order trajectories with the same reward count
            heapq.heappush(self.trajectories, (reward_count, datetime.now(), trajectory))
        logger.debug("Finished: Creating trajectories. Time taken: %f", (datetime.now() - tstart).total_seconds())

    def execute_algorithm(self, generations):
        self.population.run(self.fitness_function, generations)

    def fitness_function(self, genomes, config):
        """
        This method is called every generation.
        Create new array
        :param genomes:
        :param config:
        :return:
        """
        logger.debug("Initialising neural networks")
        nets = []
        for genome_id, genome in genomes:
            # reinitialise the agents
            network = neat.nn.FeedForwardNetwork.create(genome, config)
            neat_network = NeatEMAgent(network,
                                       props.getint('neuralnet', 'dimension'),
                                       props.getint('policy', 'num_actions'))
            nets.append((genome, neat_network))
            genome.fitness = 0

        logger.debug("Finished: Initialising neural networks")

        '''
        Collect a set of trajectories from the trajectories
        '''
        state_transitions = set()
        for i in range(len(self.trajectories)):
            _, __, trajectory = self.trajectories[i]
            state_transitions = state_transitions | set(trajectory)

        tstart = datetime.now()
        '''
        For each individual update value function using TD error and experience replay
        And update the policy parameter
        '''
        for genome, net in nets:
            experience_replay = props.getint('evaluation', 'experience_replay')
            random_state_transitions = random.sample(state_transitions, experience_replay)

            # update value function
            net.update_value_function(random_state_transitions)

            # update policy parameter
            net.update_policy_function(self.trajectories)

            # now assign fitness to each individual/genome
            # fitness is the log prob of following the best trajectory
            # I need the get action to return me the probabilities of the actions rather than a numerical action
            best_trajectory = heapq.nlargest(1, self.trajectories)
            best_trajectory_prob = 0
            total_reward, _, trajectory_state_transitions = best_trajectory[0]
            for j, state_transition in enumerate(trajectory_state_transitions):
                # calculate probability of the action probability where policy action = action
                state_features = net.get_network().activate(state_transition.get_start_state())
                _, actions_distribution = net.get_policy().get_action(state_features)
                best_trajectory_prob += np.log(actions_distribution[state_transition.get_action()])

            fitness = best_trajectory_prob
            genome.fitness = fitness

        # select K random agents to perform rollout

        num_new_trajectories = props.getint('evaluation', 'new_trajectories')

        if num_new_trajectories > len(nets):
            logger.debug("number of new trajectories is greater than the number of agents")
            num_new_trajectories = len(nets)

        logger.debug("Generating %d new trajectories", num_new_trajectories)

        max_steps = props.getint('initialisation', 'max_steps')
        step_size = props.getint('initialisation', 'step_size')
        nets_sorted = sorted(nets, key=lambda x: x[0].fitness, reverse=True) # could replace this with an in-place sort

        for i in range(num_new_trajectories):
            genome, agent = nets_sorted[i]
            # perform a rollout
            state = env.reset()
            terminal_reached = False
            steps = 0
            reward_count = 0
            new_trajectory = []
            while not terminal_reached and steps < max_steps:
                # env.render()
                state_features = agent.get_network().activate(state)
                # get recommended action and the action distribution using policy
                action, actions_distribution = agent.get_policy().get_action(state_features)
                next_state, reward, done, info = env.step(action)

                for x in range(step_size - 1):
                    if done:
                        terminal_reached = True
                        break
                    next_state, reward2, done, info = env.step(action)
                    reward += reward2

                # insert state transition to the trajectory
                state_transition = StateTransition(state, action, reward, next_state)
                new_trajectory.append(state_transition)
                reward_count += reward
                state = next_state
                steps += 1
                if done:
                    terminal_reached = True

            '''
            In MountainCar problem, the total reward is always negative. The lower the number the worse the trajectory
            Heapq keeps smallest values in the first position.
            '''
            heapq.heappush(self.trajectories, (reward_count, datetime.now(), new_trajectory))

        # strip weak trajectories from trajectory_set and add state transitions to set state_transitions
        self.trajectories = heapq.nlargest(props.getint('initialisation', 'trajectory_size'), self.trajectories)
        logger.debug("Worst Trajectory reward: %f", self.trajectories[len(self.trajectories) - 1][0])
        logger.debug("Best Trajectory reward: %f", self.trajectories[0][0])

        # save the best individual's genome
        genome, net = nets_sorted[0]
        logger.debug("Best genome: %s", genome)
        logger.debug("Best genome fitness: %f", genome.fitness)

        # Select best agent (using fitness function) and test
        self.best_agents.append((self.generation_count, genome, net))

        genome_worst, net = nets_sorted[len(nets_sorted) - 1]
        logger.debug("Worst genome fitness: %f", genome_worst.fitness)

        # save genome
        with open('best_genomes/gen-{0}-genome'.format(self.generation_count), 'wb') as f:
            pickle.dump(genome, f)

        self.generation_count += 1
        logger.debug("Finished: Generating %d new trajectories", num_new_trajectories)
        logger.debug("Completed Generation. Time taken: %f", (datetime.now() - tstart).total_seconds())


def test_best_agent(generation_count, genome, agent):
    logger.debug("Generating best agent results: %d", generation_count)
    t_start = datetime.now()

    max_steps = props.getint('initialisation', 'max_steps')
    test_episodes = props.getint('test', 'test_episodes')
    step_size = props.getint('initialisation', 'step_size')

    total_steps = 0.0
    total_rewards = 0.0
    for i in range(test_episodes):
        state = env.reset()
        terminal_reached = False
        steps = 0
        while not terminal_reached and steps < max_steps:
            env.render()
            state_features = agent.get_network().activate(state)
            action, actions_distribution = agent.get_policy().get_action(state_features)
            next_state, reward, done, info = env.step(action)

            for x in range(step_size - 1):
                if done:
                    terminal_reached = True
                    break
                next_state, reward2, done, info = env.step(action)
                reward += reward2

            total_rewards += reward
            state = next_state

            steps += 1
            if done:
                terminal_reached = True
        total_steps += steps
    average_steps_per_episodes = total_steps / test_episodes
    average_rewards_per_episodes = total_rewards / test_episodes

    # save this to file along with the generation number
    entry = [generation_count, average_steps_per_episodes, average_rewards_per_episodes]
    with open(r'agent_evaluation.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(entry)

    logger.debug("Finished: evaluating best agent. Time taken: %f", (datetime.now() - t_start).total_seconds())


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
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    gym.undo_logger_setup()
    logging.basicConfig(filename='log/debug-'+str(datetime.now())+'.log', level=logging.DEBUG)
    logger = logging.getLogger()
    logging.Formatter('[%(asctime)s] %(message)s')
    env = gym.make(args.env_id)

    # logger.debug(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
    # env.spec.tags['wrapper_config.TimeLimit.max_episode_steps'] = 200
    # logger.debug(env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps'))
    # env._max_episode_steps = 200

    # env = env.env
    logger.debug("action space: %s", env.action_space)
    logger.debug("observation space: %s", env.observation_space)


    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    # outdir = '~/tmp/neat-em-data/' + str(datetime.now())
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    local_dir = os.path.dirname(__file__)
    # load properties
    logger.debug("Loading Properties File")
    props = configparser.ConfigParser()
    prop_path = os.path.join(local_dir, 'properties/{0}/neatem_properties.ini'.format(env.spec.id))
    props.read(prop_path)
    logger.debug("Finished: Loading Properties File")

    # Load the config file, which is assumed to live in
    # the same directory as this script.

    config_path = os.path.join(local_dir, 'properties/{0}/config'.format(env.spec.id))
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    population = NeatEM(config)

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    while 1:
        try:
            # Run for 5 generations
            population.execute_algorithm(props.getint('neuralnet', 'generation'))

            # Generate test results
            outdir = 'videos/tmp/neat-em-data/{0}-{1}'.format(env.spec.id, str(datetime.now()))
            env = wrappers.Monitor(env, directory=outdir, force=True)
            for (generation_count, genome, agent) in population.best_agents:
                test_best_agent(generation_count, genome, agent)

            visualize.plot_stats(population.stats, ylog=False, view=False, filename="fitness.svg")

            mfs = sum(population.stats.get_fitness_mean()[-20:]) / 20.0
            logger.debug("Average mean fitness over last 20 generations: %f", mfs)

            mfs = sum(population.stats.get_fitness_stat(min)[-20:]) / 20.0
            logger.debug("Average min fitness over last 20 generations: %f", mfs)

            # Use the 10 best genomes seen so far
            best_genomes = population.stats.best_unique_genomes(10)

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
