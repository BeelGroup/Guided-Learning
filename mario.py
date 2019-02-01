import configparser
import retro
import neat
from neat.six_util import iteritems
import numpy as np

from matplotlib import pyplot as plt

import visualize
from inputs import *


class Mario:
    def __init__(self, env, neat_config_file, verbosity=0):

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

        self.env = env
        self.verbosity = verbosity
        self.timeout = int(self.config['NEAT']['timeout'])
        self.current_frame = self.env.reset()
        self.joystick_inputs = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]).astype(np.uint8)
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



    def get_neat_inputs(self):
        '''Uses the given frame to compute an output for each 16x16 block of pixels.
            Extracts player position and enemy positions (-1 if unavailable) from info.
                Returns: A list of inputs to be fed to the NEAT network '''
        frame = self.current_frame
        info = self.current_info

        # player_pos_x, player_pos_y, enemy_n_pos_x, enemy_n_pos_y, tile_input
        inputs = []

        # raw positions are given in (y,x with origin at the bottom right)
        player_pos = get_raw_player_pos(info)
        inputs.append(player_pos[0])
        inputs.append(player_pos[1])

        enemy_pos = get_raw_enemy_pos(info)
        for enemy in enemy_pos:
            inputs.append(enemy[0])
            inputs.append(enemy[1])

        if self.config['NEAT'].getboolean('inputs_greyscale'):
            frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
            frame.resize((frame.shape[0], frame.shape[1], 1))

        tiles = get_tiles(frame, 16, 16, info['xscrollLo'], player_pos=player_pos, radius=int(self.config['NEAT']['inputs_radius']))

        # Get an array of average values per tile (this may have 3 values for RGB or 1 for greyscale)
        tile_avg = np.mean(tiles, axis=3, dtype=np.uint16) # average across tile row
        tile_avg = np.mean(tile_avg, axis=2, dtype=np.uint16)  # average across tile col

        if self.config['NEAT'].getboolean('inputs_greyscale'):
            # the greyscale value is in all 3 positions so just get the first
            tile_inputs = tile_avg[:, :, 0:1].flatten().tolist()
        else:
            tile_inputs = tile_avg[:, :, :].flatten().tolist()

        inputs = inputs + tile_inputs

        if self.debug:
            print("[get_neat_inputs] Raw Player Pos: {}".format(player_pos))
            print("[get_neat_inputs] Raw Enemy Pos: {}".format(enemy_pos))
            if self.debug_graphs:
                print("[get_neat_inputs] Displaying tile inputs:")
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                major_ticks = np.arange(0, 256, 16)
                ax.set_xticks(major_ticks)
                ax.set_yticks(major_ticks)
                ax.imshow(stitch_tiles(tiles, 16, 16))
                plt.grid(True)
                plt.show()
        return inputs


    def run(self):

        while True:
            for genome_id, genome in list(iteritems(self.neat.population)):
                try:
                    self.current_net = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)

                    self.current_frame = self.env.reset()

                    t = 0
                    totrew = [0] * 1
                    self.timeout = int(self.config['NEAT']['timeout'])

                    while True:
                        # Evaluate the current genome
                        if t % 10 == 0:
                            if self.verbosity > 1:
                                infostr = ''
                                if self.current_info:
                                    infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in self.current_info.items()])
                                print(('t=%i' % t) + infostr)

                            if self.current_info is not None:
                                inputs = self.get_neat_inputs()
                                raw_joystick_inputs = self.current_net.activate(inputs)
                                self.joystick_inputs = [round(x) for x in raw_joystick_inputs]
                                self.env.render()

                        self.current_frame, rew, done, self.current_info = self.env.step(self.joystick_inputs)

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
