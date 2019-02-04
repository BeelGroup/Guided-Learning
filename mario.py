import configparser, datetime, time
import retro
import neat
from neat.six_util import iteritems
import numpy as np

import visualize
from inputs import *
from human_input import get_human_input

class Mario:
    def __init__(self, retro_env, neat_config_file, verbosity=0):

        print(get_human_input(retro_env))

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
        self.current_player_pos = None
        self.joystick_inputs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.uint8)
        self.current_net = None
        self.current_info = None

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


    def train_human_input(self):
        # TODO
        human_input = get_human_input(self.env)
        pass


    def get_neat_inputs(self):
        '''Uses the given frame to compute an output for each 16x16 block of pixels.
            Extracts player position and enemy positions (-1 if unavailable) from info.
                Returns: A list of inputs to be fed to the NEAT network '''

        # player_pos_x, player_pos_y, enemy_n_pos_x, enemy_n_pos_y, tile_input
        inputs = []

        # normalize
        inputs.append(self.current_player_pos[0] / self.current_frame.shape[0])
        inputs.append(self.current_player_pos[1] / self.current_frame.shape[1])

        enemy_pos = get_raw_enemy_pos(self.current_info)
        for enemy in enemy_pos:
            if enemy != (-1, -1):
                # normalize
                inputs.append(enemy[0] / self.current_frame.shape[0])
                inputs.append(enemy[1] / self.current_frame.shape[1])
            else:
                inputs.append(enemy[0])
                inputs.append(enemy[1])

        inputs = inputs + get_screen_inputs(self.current_frame, self.current_info, self.config, debug=self.debug_graphs)

        if self.debug:
            print("[get_neat_inputs] Raw Player Pos: {}".format(self.current_player_pos))
            print("[get_neat_inputs] Raw Enemy Pos: {}".format(enemy_pos))

        return inputs


    def run(self, framerate_limit):
        ''' The main loop '''
        frame_delay = 1000/framerate_limit if framerate_limit != -1 else 0

        while True:

            prev_best_fitness = None
            for genome_id, genome in list(iteritems(self.neat.population)):
                try:
                    stagnation_count = 0

                    self.current_net = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)

                    self.current_frame = self.env.reset()

                    t = 0
                    totrew = [0] * 1
                    self.timeout = int(self.config['NEAT']['timeout'])

                    frame_delay_start_time = get_epochtime_ms()
                    while True:
                        # limit loop based on frame_delay
                        while(get_epochtime_ms() < frame_delay_start_time + frame_delay):
                            pass
                        frame_delay_start_time = get_epochtime_ms()

                        # Evaluate the current genome
                        if t % 10 == 0:
                            if self.verbosity > 1:
                                infostr = ''
                                if self.current_info:
                                    infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in self.current_info.items()])
                                print(('t=%i' % t) + infostr)

                            if self.current_info is not None:
                                # raw positions are given in (y,x with origin at the bottom right)
                                self.current_player_pos = get_raw_player_pos(self.current_info)

                                inputs = self.get_neat_inputs()
                                raw_joystick_inputs = self.current_net.activate(inputs)
                                # pad None into second position
                                self.joystick_inputs = raw_joystick_inputs[:1] + [0] + raw_joystick_inputs[1:]
                                # round to 0 or 1
                                self.joystick_inputs = np.asarray([round(x) for x in self.joystick_inputs], dtype=np.uint8)

                            self.env.render()

                        self.current_frame, rew, done, self.current_info = self.env.step(self.joystick_inputs)
                        rew = [rew]

                        t += 1

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
                        if done:
                            self.env.render()
                            print("GenomeId:%s time:%i fitness:%d" % (genome_id, t, totrew[0]))
                            genome.fitness = totrew[0]
                            break

                except KeyboardInterrupt:
                    exit(0)

            # All genomes in the current population have been evaluated, get the best genome and move to the next generation
            self.neat.next_generation()
            print("Best of gen: {} -- Fitness: {!s} -- Shape: {}".format(self.neat.generation-1, self.neat.best_genome.fitness, self.neat.best_genome.size()))
            # visualise the champion
            visualize.draw_net(self.neat_config, self.neat.best_genome, view=False, filename="img/gen_{}_genome".format(
                self.neat.generation - 1))
            visualize.plot_stats(self.neat_stats)

            if prev_best_fitness is not None and prev_best_fitness == self.neat.best_genome.fitness:
                stagnation_count += 1
            else:
                prev_best_fitness = self.neat.best_genome.fitness

            if stagnation_count == 2:
                print("STAGNATION")

            # save the current state
            self.save()

