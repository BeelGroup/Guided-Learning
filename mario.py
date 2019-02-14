import configparser, datetime, time
import retro
import neat
from neat.six_util import iteritems
import numpy as np

from threading import Timer

import visualize
from inputs import *
from human_input import get_human_input
from keras_example import train_single_shot, keras2neat

class Mario:
    def __init__(self, retro_env, neat_config_file, verbosity=0):

        #print(get_human_input(retro_env))

        # Load configuration.
        self.config = configparser.ConfigParser()
        self.config.read('mario.config')

        # get the NEAT config
        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, neat_config_file)

        ''' 
        # TODO: Allow dynamic input length based on config['NEAT'].getboolean('inputs_greyscale')
        # set the number of genome inputs dynamically based on RGB/Greyscale
        self.num_inputs = (16*16*(int(self.config['NEAT']['inputs_radius']) ** 2)) + 25 + 4
        if not self.config['NEAT'].getboolean('inputs_greyscale'):
            self.num_inputs *= 3
        self.neat_config.genome_config['num_inputs'] = self.num_inputs
        '''

        self.env = retro_env
        self.verbosity = verbosity
        self.timeout = int(self.config['NEAT']['timeout'])
        self.current_frame = self.env.reset()
        self.current_net = None
        self.current_info = None
        self.frame_delay = None
        self.current_best_genome = None
        self.start_state = self.env.initial_state
        self.human_start_state = None
        self.taught_responses = [] # list of tuples (input, model)

        self.debug = self.config['DEFAULT'].getboolean('debug')
        self.debug_graphs = self.config['DEFAULT'].getboolean('debug_graphs')

        # Create the population, which is the top-level object for a NEAT run.
        print("Generating NEAT population..")
        self.neat = neat.Population(self.neat_config)

        # Add a stdout reporter to show progress in the terminal.
        self.neat_stats = neat.StatisticsReporter()
        self.neat.add_reporter(self.neat_stats)

        print("Population Generated.")


    def set_env(self, env):
        ''' Used to set the Retro env after loading from picked file. '''
        self.env = env


    def save(self):
        ''' Used to save the current state of self '''
        env = self.env
        self.env = None
        save_state(self, "saves/gen_{}.bkup".format(self.neat.generation - 1))
        self.env = env

    def ask_for_help(self):
        print("[ask_for_help] STAGNATION - Asking for help..")

        # take input here to verify help being given

        # play the best genome up to just before it dies
        self.evaluate_genome(self.current_best_genome, fps=30, human_intervention=True)
        # save the current state
        self.human_start_state = self.env.em.get_state()
        self.env.initial_state = self.human_start_state

        # Take the human input
        human_io = get_human_input(self.env) # returns A list of tuples: (obs, info, action)
        if len(human_io) == 0:
            print("[ask_for_help] No input received. Continuing.")
        else:
            print("[ask_for_help] Number of human_io samples: {}".format(len(human_io)))

            inputs = np.asarray([get_network_inputs(h_io[0], h_io[1], self.config) for h_io in human_io])
            # remove the second position control
            expected_outputs = np.asarray([np.append(h_io[2][:1], h_io[2][2:]) for h_io in human_io])

            model = train_single_shot(inputs, expected_outputs)

            neat_genome = keras2neat(self.neat_config, 'model.h5', new_genome_key=str(len(self.taught_responses)))

            ## TEST
            print("[ask_for_help] KERAS_MODEL_PREDICT: {}".format(model.predict(inputs)))
            neat_genome_ff = neat.nn.FeedForwardNetwork.create(neat_genome, self.neat_config)
            print("[ask_for_help] NEAT_GENOME_PREDICT: {}".format(neat_genome_ff.activate(inputs[0])))

            self.taught_responses.append((inputs, neat_genome))

            print("[ask_for_help] Evaluating the trained network..")
            self.evaluate_genome(neat_genome, fps=30)

        # reset the start state
        self.env.initial_state = self.start_state


    def get_taught_response(self):
        if len(self.taught_responses) > 0:
            current_inputs = np.asarray(get_network_inputs(self.current_frame, self.current_info, self.config))
            for mem_inputs, mem in self.taught_responses:
                if abs(np.sum(current_inputs - mem_inputs)) < 0.02:
                    print("[get_taught_response] Triggering TRN..")
                    print("[get_taught_response] Diff: {}".format(abs(np.sum(current_inputs - mem_inputs))))
                    print("[get_taught_response] inputs: {}".format(current_inputs))
                    print("[get_taught_response] inputs.shape: {}".format(current_inputs.shape))
                    mem_net = neat.nn.FeedForwardNetwork.create(mem, self.neat_config)
                    raw_joystick_inputs = mem_net.activate(current_inputs)

                    print("[get_taught_response] TRN output: {}".format(raw_joystick_inputs))
                    # pad 0 into second position
                    joystick_inputs = raw_joystick_inputs[:1] + [0] + raw_joystick_inputs[1:]
                    # round outputs to 0 or 1
                    joystick_inputs = np.asarray([round(x) for x in joystick_inputs], dtype=np.uint8)

                    print("[get_taught_response] TRN joystick_inputs: {}".format(joystick_inputs))

                    return joystick_inputs
        else:
            return None

    def evaluate_genome(self, genome, fps=-1, human_intervention=False):
        '''
        :param genome: The genome to evaluate
        :param fps: Limits to the given fps
        :param human_intervention: Signals that human is giving help, if true then genome.fitness should have a value
        :return: None
        '''

        self.frame_delay = 1000 / fps if fps != -1 else 0

        joystick_inputs = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1]).astype(np.uint8)

        try:
            self.current_net = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)

            self.current_frame = self.env.reset()

            t = 0
            totrew = [0] * 1
            self.timeout = int(self.config['NEAT']['timeout'])

            frame_delay_start_time = get_epochtime_ms()
            while True:

                if fps != -1:
                    # limit loop based on frame_delay
                    while (get_epochtime_ms() < frame_delay_start_time + self.frame_delay):
                        pass
                    frame_delay_start_time = get_epochtime_ms()

                if self.current_info:
                    # Search for taught response memories
                    tr = self.get_taught_response()
                    if tr is not None:
                        joystick_inputs = tr
                        t = 1

                if t % 10 == 0 or self.frame_delay != 0:
                    if self.current_info:
                        inputs = get_network_inputs(self.current_frame, self.current_info, self.config)
                        raw_joystick_inputs = self.current_net.activate(inputs)
                        # pad 0 into second position
                        joystick_inputs = raw_joystick_inputs[:1] + [0] + raw_joystick_inputs[1:]
                        # round outputs to 0 or 1
                        joystick_inputs = np.asarray([round(x) for x in joystick_inputs], dtype=np.uint8)

                    # render every 10th frame in simulation or every frame if fps limit is set
                    self.env.render()

                self.current_frame, rew, done, self.current_info = self.env.step(joystick_inputs)

                t += 1

                rew = [rew]
                for i, r in enumerate(rew):
                    totrew[i] += r
                    if r > 0:
                        self.timeout = int(self.config['NEAT']['timeout'])
                    else:
                        self.timeout -= 1
                        if self.timeout < 0:
                            done = True
                    if self.verbosity > 1:
                        if r > 0:
                            print('t=%i p=%i got reward: %g, current reward: %g' % (t, i, r, totrew[i]))
                        if r < 0:
                            print('t=%i p=%i got penalty: %g, current reward: %g' % (t, i, r, totrew[i]))

                if human_intervention:
                    # we are playing the best genome up to just before the stagnation point
                    if totrew[0] > genome.fitness-100:
                        break

                if done:
                    self.env.render()
                    print("GenomeId:%s time:%i fitness:%d" % (genome.key, t, totrew[0]))
                    genome.fitness = totrew[0]
                    break

        except KeyboardInterrupt:
            exit(0)

    def run(self, fps=-1, gen_stats=True):
        ''' The main loop '''

        prev_best_fitness = None
        stagnation_count = 0

        while True:
            for genome_id, genome in list(iteritems(self.neat.population)):
                self.evaluate_genome(genome, fps=fps)

            # All genomes in the current population have been evaluated, get the best genome and move to the next generation
            self.neat.next_generation()
            self.current_best_genome = self.neat.best_genome
            print("Best of gen: {} -- Fitness: {!s} -- Shape: {}".format(self.neat.generation-1, self.neat.best_genome.fitness, self.neat.best_genome.size()))

            if prev_best_fitness is not None and prev_best_fitness <= self.neat.best_genome.fitness:
                stagnation_count += 1
            else:
                prev_best_fitness = self.neat.best_genome.fitness

            if stagnation_count > 1:
                stagnation_count = 0
                self.ask_for_help()

            if gen_stats:
                # visualise the champion
                visualize.draw_net(self.neat_config, self.neat.best_genome, view=False,
                                   filename="img/gen_{}_genome".format(
                                       self.neat.generation - 1))
                visualize.plot_stats(self.neat_stats)
                # save the current state
                self.save()
            else:
                print("Statistics and backup are disabled.")

