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
        self.taught_responses = [] # list of tuples (input, model, count, best_fitness)

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
        self.evaluate_genome(self.current_best_genome, human_intervention=True)
        # save the current state
        self.human_start_state = self.env.em.get_state()
        self.env.initial_state = self.human_start_state


        # Take the human input (A list of tuples: (obs, info, action), count of each action)
        human_io, human_io_count = get_human_input(self.env)
        if len(human_io) == 0:
            print("[ask_for_help] No input received. Continuing.")
        else:
            print("[ask_for_help] Number of human_io samples: {}".format(len(human_io)))
            print("[ask_for_help] Len of each sample: {}".format(human_io_count))

            assert(len(human_io) == 1 and len(human_io_count) == 1, "Only single shot TRMs are supported!!!")

            inputs = np.asarray([get_network_inputs(h_io[0], h_io[1], self.config) for h_io in human_io])
            # remove the second position control
            expected_outputs = np.asarray([np.append(h_io[2][:1], h_io[2][2:]) for h_io in human_io])

            model = train_single_shot(inputs, expected_outputs)

            neat_genome = keras2neat(self.neat_config, 'model.h5', new_genome_key=str(len(self.taught_responses)))

            ## TEST
            print("[ask_for_help] KERAS_MODEL_PREDICT: {}".format(model.predict(inputs)))
            neat_genome_ff = neat.nn.FeedForwardNetwork.create(neat_genome, self.neat_config)
            print("[ask_for_help] NEAT_GENOME_PREDICT: {}".format(neat_genome_ff.activate(inputs[0])))

            tr_score_replacement_threshold = 10
            replace_trm = -1
            for i, tr in enumerate(self.taught_responses):
                if abs(tr[3]- self.neat.best_genome.fitness) < tr_score_replacement_threshold:
                    replace_trm = i
                    print("[ask_for_help] Replacing TRM: {}".format(i))
                    print("[ask_for_help] Number of TRMs: {}".format(len(self.taught_responses)))
                    break
            if replace_trm == -1:
                self.taught_responses.append((inputs, neat_genome, human_io_count, self.neat.best_genome.fitness))
            else:
                self.taught_responses[i] = (inputs, neat_genome, human_io_count, self.neat.best_genome.fitness)

            #print("[ask_for_help] Evaluating the trained network..")
            #self.evaluate_genome(neat_genome, fps=30)

        # reset the start state
        self.env.initial_state = self.start_state


    def get_taught_response(self):
        if len(self.taught_responses) > 0:
            current_inputs = np.asarray(get_network_inputs(self.current_frame, self.current_info, self.config))
            for mem_inputs, mem, count, fitness in self.taught_responses:
                if abs(np.sum(current_inputs - mem_inputs)) < 0.015: # 0.02 FOR NORMALIZED TILES
                    print()
                    print("[get_taught_response] Triggering TRN..")
                    print("[get_taught_response] Diff: {}".format(abs(np.sum(current_inputs - mem_inputs))))
                    #print("[get_taught_response] inputs: {}".format(current_inputs))
                    #print("[get_taught_response] inputs.shape: {}".format(current_inputs.shape))
                    mem_net = neat.nn.FeedForwardNetwork.create(mem, self.neat_config)
                    raw_joystick_inputs = mem_net.activate(current_inputs)

                    #print("[get_taught_response] TRN output: {}".format(raw_joystick_inputs))
                    # pad 0 into second position
                    joystick_inputs = raw_joystick_inputs[:1] + [0] + raw_joystick_inputs[1:]
                    # round outputs to 0 or 1
                    joystick_inputs = np.asarray([round(x) for x in joystick_inputs], dtype=np.uint8)

                    print("[get_taught_response] TRN joystick_inputs: {}".format(joystick_inputs))
                    print("[get_taught_response] TRN_count: {}".format(count))
                    print()

                    return (joystick_inputs, count)
                #else:
                #    print("Diff: {}".format(abs(np.sum(current_inputs - mem_inputs))))
        return (None, None)

    def evaluate_genome(self, genome, fps=-1, human_intervention=False):
        '''
        :param genome: The genome to evaluate
        :param fps: Limits to the given fps
        :param human_intervention: Signals that human is giving help, if true then genome.fitness should have a value
        :return: None
        '''

        self.frame_delay = 1000 / fps if fps != -1 else 0

        joystick_inputs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.uint8)

        try:
            self.current_net = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)

            self.current_frame = self.env.reset()

            t = 0
            totrew = [0] * 1
            self.timeout = int(self.config['NEAT']['timeout'])

            TRM_count = 0

            frame_delay_start_time = get_epochtime_ms()

            while True:

                if fps != -1:
                    # limit loop based on frame_delay
                    while (get_epochtime_ms() < frame_delay_start_time + self.frame_delay):
                        pass
                    frame_delay_start_time = get_epochtime_ms()

                if self.current_info:
                    # Search for taught response memories
                    tr, count = self.get_taught_response()
                    if tr is not None:
                        joystick_inputs = tr
                        TRM_count = count[0]+10 # SINGLE SHOT ONLY

                if (t % 10 == 0 or self.frame_delay != 0) and TRM_count==0:
                    if self.current_info:
                        inputs = get_network_inputs(self.current_frame, self.current_info, self.config)
                        raw_joystick_inputs = self.current_net.activate(inputs)
                        # pad 0 into second position
                        joystick_inputs = raw_joystick_inputs[:1] + [0] + raw_joystick_inputs[1:]
                        # round outputs to 0 or 1
                        joystick_inputs = np.asarray([round(x) for x in joystick_inputs], dtype=np.uint8)

                if t % 10 == 0 or self.frame_delay != 0:
                    # render every 10th frame in simulation or every frame if fps limit is set
                    self.env.render()

                if TRM_count > 0:
                    TRM_count -= 1

                self.current_frame, rew, done, self.current_info = self.env.step(joystick_inputs)

                t += 1

                done = self.determine_done_condition(rew, totrew)

                if human_intervention:
                    # we are playing the best genome up to just before the stagnation point
                    if totrew[0] > genome.fitness-50:
                        break

                if done:
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

            if prev_best_fitness is not None and prev_best_fitness == self.neat.best_genome.fitness:
                stagnation_count += 1
            else:
                prev_best_fitness = self.neat.best_genome.fitness
                stagnation_count = 0

            if stagnation_count > 2:
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

    def determine_done_condition(self, rew, totrew):
        done = False
        rew = [rew]
        for i, r in enumerate(rew):
            totrew[i] += r
            if r > 0:
                self.timeout = int(self.config['NEAT']['timeout'])
            else:
                self.timeout -= 1
                if self.timeout < 0:
                    done = True
                    break
        return done
