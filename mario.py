import configparser
import retro
import neat
from neat.six_util import iteritems
import numpy as np

from inputs import get_neat_inputs


class Mario:
    def __init__(self, env, neat_config_file, verbosity=0):

        # Load configuration.
        self.mario_config = configparser.ConfigParser()
        self.mario_config.read('mario.config')

        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, neat_config_file)

        self.env = env
        self.verbosity = verbosity
        self.timeout = int(self.mario_config['NEAT']['timeout'])
        self.current_frame = self.env.reset()
        self.joystick_inputs = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]).astype(np.uint8)
        self.current_net = None
        self.current_info = None

        # Create the population, which is the top-level object for a NEAT run.
        print("Generating NEAT population..")
        self.neat = neat.Population(self.neat_config)
        print("Population Generated.")


    def run(self):
        while True:
            for genome_id, genome in list(iteritems(self.neat.population)):
                try:
                    self.current_net = neat.nn.FeedForwardNetwork.create(genome, self.neat_config)

                    self.current_frame = self.env.reset()

                    t = 0
                    totrew = [0] * 1
                    self.timeout = int(self.mario_config['NEAT']['timeout'])

                    while True:
                        # Evaluate the current genome
                        if t % 10 == 0:
                            print()
                            if self.verbosity > 1:
                                infostr = ''
                                if self.current_info:
                                    infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in self.current_info.items()])
                                print(('t=%i' % t) + infostr)

                            if self.current_info is not None:
                                inputs = get_neat_inputs(self.current_frame, self.current_info, self.mario_config['NEAT'])
                                # TODO: Allow dynamic input length based on config['NEAT'].getboolean('inputs_greyscale')
                                # self.joystick_inputs = self.current_net.activate(inputs)

                                self.env.render()

                        self.current_frame, rew, done, self.current_info = self.env.step(self.joystick_inputs)

                        t += 1

                        rew = [rew]
                        for i, r in enumerate(rew):
                            totrew[i] += r
                            if r > 0:
                                self.timeout = int(self.mario_config['NEAT']['timeout'])
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
            print("Best of gen: ", self.neat.generation - 1, "\nFitness: {!s}".format(self.neat.best_genome.fitness))