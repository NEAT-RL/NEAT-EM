from __future__ import print_function

import multiprocessing
import os
import pickle
import argparse
import logging
import random
import uuid
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
from time import sleep


class NeatEM(object):
    def __init__(self, config):
        self.num_actions = props.getint('policy', 'num_actions')
        self.max_steps = props.getint('train', 'max_steps')
        self.step_size = props.getint('train', 'step_size')
        self.iterations = props.getint('train', 'iterations')
        self.num_trajectories = props.getint('trajectory', 'trajectory_size')
        self.experience_replay = props.getint('evaluation', 'experience_replay')
        self.generation_num = 0
        # Initialise the neat population and configurations
        pop = neat.Population(config)
        self.stats = neat.StatisticsReporter()
        pop.add_reporter(self.stats)
        pop.add_reporter(neat.StdOutReporter(True))
        self.population = pop
        self.trajectories = []
        self.__initialise_trajectories()

    def __initialise_trajectories(self):
        """
        Initialise trajectories of size: 'size'.
        Each trajectory we store N number of state transitions. (state, reward, next state, action)
        :param:
        :return:
        """
        logger.debug("Creating trajectories for first time...")
        t_start = datetime.now()
        num_actions = self.num_actions
        for x in range(self.num_trajectories):
            self.trajectories.append(NeatEM.initialise_trajectory(num_actions))

        self.trajectories.sort(reverse=True)
        logger.debug("Finished: Creating trajectories. Time taken: %f", (datetime.now() - t_start).total_seconds())

    @staticmethod
    def initialise_trajectory(num_actions):
        max_steps = props.getint('train', 'max_steps')
        step_size = props.getint('train', 'step_size')
        state_starts = []
        state_ends = []
        rewards = []
        actions = []
        state = env.reset()
        terminal_reached = False
        steps = 0
        total_reward = 0
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

            action_array = np.zeros((num_actions,))
            action_array[action] = 1
            state_starts.append(state)
            state_ends.append(next_state)
            rewards.append(reward)
            actions.append(action_array)

            total_reward += reward
            state = next_state
            steps += 1
            if done:
                terminal_reached = True

        return total_reward, uuid.uuid4(), state_starts, state_ends, actions, rewards

    def execute_algorithm(self, generations):
        self.population.run(self.execute_generation, generations)

    def execute_generation(self, genomes, config):
        # Firstly, initialise the agents and their theano equations. Because this is slower, we do this in parallel
        t_start = datetime.now()
        results = []
        if allow_multiprocessing:
            create_agents = [pool.apply_async(NeatEM.create_agent) for i in range(len(genomes))]
            results = [create_agent.get() for create_agent in create_agents]
        else:
            for i in range(len(genomes)):
                results.append(NeatEM.create_agent())
        # Second, add network as feature and genome id
        agents = []
        genome_dict = {}
        for i, (gid, genome) in enumerate(genomes):
            genome_dict[gid] = genome
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            results[i].create_feature(net, gid)
            agents.append(results[i])

        # Thirdly, Learn the policy parameters:
        for i in range(self.iterations):
            logger.debug("Running iteration: %d", i)
            self.fitness_function(agents)
            if i % (50 - 2) == 0:
                # Sleep for testing the values etc.
                sleep(10)

        # Fourth, calculate the fitness of each agent based on the best trajectories

        best_trajectories = heapq.nlargest(5, self.trajectories)
        best_start_states = []
        best_actions = []

        for i, (_, _, state_starts, state_ends, actions, rewards) in enumerate(best_trajectories):
            best_start_states += state_starts
            best_actions += actions

        best_agent = None
        for agent in agents:
            best_trajectory_prob = agent.calculate_agent_fitness(best_start_states, best_actions)
            agent.fitness = best_trajectory_prob
            logger.debug("Best trajectory fitness: %f", best_trajectory_prob)
            genome_dict[agent.genome_id].fitness = best_trajectory_prob
            if best_agent is None or best_agent.fitness < agent.fitness:
                best_agent = agent

        # select agent with the best fitness and generate result
        test_best_agent(self.generation_num, best_agent)
        self.generation_num += 1

        logger.debug("Completed Generation. Time taken: %f", (datetime.now() - t_start).total_seconds())


    @staticmethod
    def create_agent():
        dimension = props.getint('feature', 'dimension')
        num_actions = props.getint('policy', 'num_actions')
        agent = NeatEMAgent(None, None, dimension,
                            num_actions)
        return agent

    def fitness_function(self, agents):
        # strip weak trajectories from trajectory_set
        self.trajectories = heapq.nlargest(self.num_trajectories, self.trajectories)

        logger.debug("Worst Trajectory reward: %f", self.trajectories[len(self.trajectories) - 1][0])
        logger.debug("Best Trajectory reward: %f", self.trajectories[0][0])

        all_state_starts = []
        all_state_ends = []
        all_rewards = []
        all_actions = []

        for j, (_, _, state_starts, state_ends, actions, rewards) in enumerate(self.trajectories):
            all_actions += actions
            all_rewards += rewards
            all_state_starts += state_starts
            all_state_ends += state_ends

        len_state_transitions = len(all_state_starts)
        random_indexes = random.sample(range(0, len_state_transitions), self.experience_replay if len_state_transitions > self.experience_replay else len_state_transitions)

        if allow_multiprocessing:
            agent_params_updates = [pool.apply_async(NeatEM.update_agent_params, args=(agent, random_indexes, all_state_starts, all_state_ends, all_actions, all_rewards))
                                      for agent in agents]

            agents = [update.get() for update in agent_params_updates]
        else:
            for agent in agents:
                NeatEM.update_agent_params(agent, random_indexes, all_state_starts, all_state_ends, all_actions, all_rewards)

        # generate 2 new trajectories using the best agent
        best_trajectories = self.trajectories[:5]
        best_start_states = []
        best_actions = []

        for k, (_, _, state_starts, state_ends, actions, rewards) in enumerate(best_trajectories):
            best_start_states += state_starts
            best_actions += actions

        best_agent = None
        for agent in agents:
            best_trajectory_prob = agent.calculate_agent_fitness(best_start_states, best_actions)
            agent.fitness = best_trajectory_prob
            # print(best_trajectory_prob)
            if best_agent is None or best_agent.fitness < agent.fitness:
                best_agent = agent

        print(best_agent.fitness)
        # print(best_agent.omega.get_value())
        num_actions = self.num_actions

        self.trajectories += self.generate_new_trajectory(best_agent, num_actions)
        self.trajectories.sort(reverse=True)

    @staticmethod
    def update_agent_params(agent, random_indexes, all_state_starts, all_state_ends, all_actions, all_rewards):
        phi = []
        phi_new = []

        for i in range(len(all_state_starts)):
            phi.append(agent.feature.phi(all_state_starts[i]))
            phi_new.append(agent.feature.phi(all_state_ends[i]))

        # value_function_phi = []
        # value_function_phi_new = []
        # value_function_rewards = []
        # for index in random_indexes:
        #     value_function_phi.append(phi[index])
        #     value_function_phi_new.append(phi_new[index])
        #     value_function_rewards.append(all_rewards[index])

        agent.update_value_function_theano(phi, phi_new, all_rewards)
        agent.update_policy_function_theano(phi, phi_new, all_actions, all_rewards)
        return agent

    @staticmethod
    def generate_new_trajectory(agent, num_actions):
        max_steps = props.getint('train', 'max_steps')
        step_size = props.getint('train', 'step_size')
        best_average_reward = props.getint('trajectory', 'best_average_reward')
        # Repeat x number of times (5 or 10)
        new_trajectories = []
        for x in range(2):
            state_starts = []
            state_ends = []
            rewards = []
            actions = []
            # perform a rollout
            state = env.reset()
            terminal_reached = False
            steps = 0
            total_reward = 0
            while not terminal_reached and steps < max_steps:
                # env.render()
                state_features = agent.feature.phi(state)
                # get recommended action and the action distribution using policy
                action, actions_distribution = agent.get_policy().get_action(state_features)
                next_state, reward, done, info = env.step(action)

                for x in range(step_size - 1):
                    if done:
                        terminal_reached = True
                        break
                    next_state, reward2, done, info = env.step(action)
                    reward += reward2

                action_array = np.zeros((num_actions,))
                action_array[action] = 1
                state_starts.append(state)
                state_ends.append(next_state)
                rewards.append(reward)
                actions.append(action_array)

                total_reward += reward
                state = next_state
                steps += 1
                if done:
                    terminal_reached = True
            print(total_reward)
            new_trajectories.append((total_reward, uuid.uuid4(), state_starts, state_ends, actions, rewards))

        # Calculate the average of new trajectories and if its better then the best average then save policy parameters of agent
        average_reward = 0
        for i in range(len(new_trajectories)):
            average_reward += new_trajectories[i][0]

        average_reward /= len(new_trajectories)

        if average_reward >= best_average_reward:
            logger.debug("Will be using KL divergence")
            agent.get_policy().check_kl_divergence = True

        return new_trajectories


def test_best_agent(generation_num, agent):
    t_start = datetime.now()

    max_steps = props.getint('test', 'max_steps')
    test_episodes = props.getint('test', 'test_episodes')
    step_size = props.getint('test', 'step_size')

    total_steps = 0.0
    total_rewards = 0.0
    # agent.get_policy().set_policy_parameters(agent.best_policy_parameters)
    for i in range(test_episodes):
        state = env.reset()
        terminal_reached = False
        steps = 0
        while not terminal_reached and steps < max_steps:
            if display_game:
                env.render()
            state_features = agent.feature.phi(state)
            action, actions_distribution = agent.get_policy().get_action(state_features)
            state, reward, done, info = env.step(action)
            total_rewards += reward

            for x in range(step_size - 1):
                if done:
                    terminal_reached = True
                    break
                state, reward, done, info = env.step(action)
                total_rewards += reward

            steps += 1
            if done:
                terminal_reached = True

        total_steps += steps
    average_steps_per_episodes = total_steps / test_episodes
    average_rewards_per_episodes = total_rewards / test_episodes

    # save this to file along with the generation number
    entry = [generation_num, average_steps_per_episodes, average_rewards_per_episodes]
    with open(r'agent_evaluation-{0}.csv'.format(time), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(entry)

    logger.debug("Finished: evaluating best agent. Time taken: %f", (datetime.now() - t_start).total_seconds())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    parser.add_argument('--display', nargs='?', default='false', help='Show display of game. true or false')
    parser.add_argument('--threads', nargs='?', default='0', help='Number of threads to use. 0 means no threads')

    args = parser.parse_args()

    gym.undo_logger_setup()
    time = datetime.now().strftime("%Y%m%d-%H:%M:%S")
    logging.basicConfig(filename='log/debug-{0}.log'.format(time),
                        level=logging.DEBUG, format='[%(asctime)s] %(message)s')
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)

    env = gym.make(args.env_id)

    logger.debug("action space: %s", env.action_space)
    logger.debug("observation space: %s", env.observation_space)

    # load properties file
    local_dir = os.path.dirname(__file__)
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

    # initialise experiment
    # pool = multiprocessing.Pool()
    # max ==> processes = None
    # 0 ==> no multiprocessing
    # > 0 ==> use as processes
    processes = None
    allow_multiprocessing = True
    if args.threads == '0':
        allow_multiprocessing = False
    elif not args.threads == 'max':
        processes = int(args.threads)

    print(allow_multiprocessing)
    if allow_multiprocessing:
        pool = multiprocessing.Pool(processes=processes)
    experiment = NeatEM(config)

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    display_game = True if args.display == 'true' else False
    try:
        # Run for X generations
        experiment.execute_algorithm(props.getint('neuralnet', 'generation'))

        # Generate test results
        # outdir = 'videos/tmp/neat-em-data/{0}-{1}'.format(env.spec.id, str(datetime.now()))
        # env = wrappers.Monitor(env, directory=outdir, force=True)
        # test_best_agent(population.agent)

        visualize.plot_stats(experiment.stats, ylog=False, view=False, filename="fitness.svg")

        mfs = sum(experiment.stats.get_fitness_mean()[-20:]) / 20.0
        logger.debug("Average mean fitness over last 20 generations: %f", mfs)

        mfs = sum(experiment.stats.get_fitness_stat(min)[-20:]) / 20.0
        logger.debug("Average min fitness over last 20 generations: %f", mfs)

        # Use the 10 best genomes seen so far
        # best_genomes = experiment.stats.best_unique_genomes(10)

        # save_best_genomes(best_genomes, True)

    except KeyboardInterrupt:
        logger.debug("User break.")
        # save the best neural network or save top 5?
        # best_genomes = population.stats.best_unique_genomes(5)

        # save_best_genomes(best_genomes, False)
    finally:
        env.close()
        if allow_multiprocessing:
            pool.terminate()

    # Upload to the scoreboard. We could also do this from another
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
