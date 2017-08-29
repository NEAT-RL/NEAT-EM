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
        self.num_trajectories = props.getint('trajectory', 'trajectory_size')
        self.best_trajectory_reward = props.getint('trajectory', 'best_trajectory_reward')
        self.experience_replay = props.getint('evaluation', 'experience_replay')
        self.policy_state_transitions = props.getint('evaluation', 'num_policy_state_transitions')
        self.generation_num = 0
        pop = neat.Population(config)
        self.stats = neat.StatisticsReporter()
        pop.add_reporter(self.stats)
        pop.add_reporter(neat.StdOutReporter(True))
        self.population = pop
        self.best_agents = []
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

        results = [pool.apply_async(self.initialise_trajectory) for x in range(self.num_trajectories)]
        results = [trajectory.get() for trajectory in results]
        self.trajectories = results
        logger.debug("Finished: Creating trajectories. Time taken: %f", (datetime.now() - t_start).total_seconds())

    @staticmethod
    def initialise_trajectory():
        max_steps = props.getint('train', 'max_steps')
        step_size = props.getint('train', 'step_size')
        trajectory = []
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

            state_transition = StateTransition(state, action, reward, next_state)
            # insert state transition to the trajectory
            trajectory.append(state_transition)
            total_reward += reward
            state = next_state
            steps += 1
            if done:
                terminal_reached = True

        return total_reward, uuid.uuid4(), trajectory

    def execute_algorithm(self, generations):
        self.population.run(self.fitness_function, generations)

    def fitness_function(self, genomes, config):
        """
        Generate trajectory.
        Insert into Trajectories.
        Select best trajectory and perform policy update 
        :return: 
        """
        t_start = datetime.now()
        agents = []
        dimension = props.getint('neuralnet', 'dimension')
        num_actions = props.getint('policy', 'num_actions')
        for gid, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            agent = NeatEMAgent(net, dimension,
                                num_actions)
            agents.append((genome, agent))

        iterations = props.getint('train', 'iterations')
        greedy = False
        for i in range(iterations):

            # generate new trajectories Using all of the agents. Parallelise this
            new_trajectories = [pool.apply_async(self.generate_new_trajectory, args=(agent,)) for genome, agent in
                                agents]
            results = [new_trajectory.get() for new_trajectory in new_trajectories]
            self.trajectories += results

            # strip weak trajectories from trajectory_set
            self.trajectories = heapq.nlargest(self.num_trajectories, self.trajectories)

            logger.debug("Worst Trajectory reward: %f", self.trajectories[len(self.trajectories) - 1][0])
            logger.debug("Best Trajectory reward: %f", self.trajectories[0][0])

            if not greedy and self.trajectories[0][0] >= self.best_trajectory_reward:
                # Found the best possible trajectory so now turn policies into greedy one
                for genome, agent in agents:
                    agent.policy.is_greedy = True
                greedy = True

            # Collect set of state transitions
            state_transitions = set()
            for j in range(len(self.trajectories)):
                state_transitions = state_transitions | set(self.trajectories[j][2])

            random_state_transitions = random.sample(state_transitions, self.experience_replay) if len(
                state_transitions) > self.experience_replay else state_transitions

            # update value function
            value_function_updates = [pool.apply_async(agent.update_value_function, args=(random_state_transitions,))
                                      for genome, agent in agents]
            [update.get() for update in value_function_updates]

            # update policy parameters for all agents. How to parallelise this?
            random_state_transitions = random.sample(state_transitions, self.policy_state_transitions) if len(
                state_transitions) > self.policy_state_transitions else state_transitions

            policy_function_updates = [
                pool.apply_async(agent.update_policy_function, args=(random_state_transitions, state_transitions,))
                for genome, agent in agents]
            [update.get() for update in policy_function_updates]

        # calculate the fitness of each agent based on the best trajectory
        # after every x iterations. Test the agent - using the best policy parameters of each agent.
        # now assign fitness to each individual/genome
        # fitness is the log prob of following the best trajectory

        best_trajectory = heapq.nlargest(1, self.trajectories)[0]
        best_agent = None
        for genome, agent in agents:
            best_trajectory_prob = 0
            total_reward, _, trajectory_state_transitions = best_trajectory
            for j, state_transition in enumerate(trajectory_state_transitions):
                # calculate probability of the action where policy action = action
                state_features = agent.feature.phi(state_transition.get_start_state())
                _, actions_distribution = agent.get_policy().get_action(state_features)
                best_trajectory_prob += np.log(actions_distribution[state_transition.get_action()])

            genome.fitness = best_trajectory_prob
            agent.fitness = best_trajectory_prob
            if best_agent is None or best_agent.fitness > agent.fitness:
                best_agent = agent

        logger.debug("Best agent fitness: %f", best_agent.fitness)
        # select agent with the lowest fitness and generate result
        test_best_agent(self.generation_num, best_agent)

        logger.debug("Completed Generation. Time taken: %f", (datetime.now() - t_start).total_seconds())
        self.generation_num += 1

    @staticmethod
    def update_value_functions(agent, state_tranisitions):
        agent.update_value_function(state_tranisitions)

    @staticmethod
    def generate_new_trajectory(agent):
        max_steps = props.getint('train', 'max_steps')
        step_size = props.getint('train', 'step_size')

        # perform a rollout
        state = env.reset()
        terminal_reached = False
        steps = 0
        total_reward = 0
        new_trajectory = []
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

            # insert state transition to the trajectory
            state_transition = StateTransition(state, action, reward, next_state)
            new_trajectory.append(state_transition)
            total_reward += reward
            state = next_state
            steps += 1
            if done:
                terminal_reached = True

        # if total_reward > max_total_reward of policy then save policy parameters.
        if agent.max_total_reward < total_reward:
            # save policy parameters
            agent.best_policy_parameters = agent.get_policy().get_policy_parameters()
            agent.max_total_reward = total_reward

        return total_reward, uuid.uuid4(), new_trajectory


def test_best_agent(generation_num, agent):
    t_start = datetime.now()

    max_steps = props.getint('test', 'max_steps')
    test_episodes = props.getint('test', 'test_episodes')
    step_size = props.getint('test', 'step_size')

    total_steps = 0.0
    total_rewards = 0.0
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

            for x in range(step_size - 1):
                if done:
                    terminal_reached = True
                    break
                state, reward2, done, info = env.step(action)
                reward += reward2

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
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    parser.add_argument('display', nargs='?', default='false', help='Show display of game. true or false')
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
    pool = multiprocessing.Pool()
    experiment = NeatEM(config)

    # Run until the winner from a generation is able to solve the environment
    # or the user interrupts the process.
    display_game = True if args.display == 'true' else False
    try:
        # Run for 5 generations
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
        pool.terminate()

    # Upload to the scoreboard. We could also do this from another
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)
