#!/usr/bin/env python

import argparse, configparser, os, sys, time, math
import retro

import numpy as np

from skimage.measure import block_reduce
from skimage.segmentation import felzenszwalb, slic

import neat
from neat.six_util import iteritems
import visualize

from inputs import get_neat_inputs


parser = argparse.ArgumentParser()
parser.add_argument('--game', default='SuperMarioBros-Nes', help='the name or path for the game to run')
parser.add_argument('--state', default='Level1-1', help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario.json', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
args = parser.parse_args()

mario_config = configparser.ConfigParser()
mario_config.read('mario.config')

env = retro.make(args.game, args.state or retro.State.DEFAULT, scenario=args.scenario, record=args.record, players=args.players)

verbosity = args.verbose - args.quiet

TIMEOUT_DEFAULT = int(mario_config['NEAT']['timeout'])


def main(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    print("Generating NEAT population..")
    p = neat.Population(config)
    print("Population Generated.")

    info = None
    inputs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.uint8)
    while True:
        for genome_id, genome in list(iteritems(p.population)):
            try:
                net = neat.nn.FeedForwardNetwork.create(genome, config)

                ob = env.reset()

                t = 0
                timeout = TIMEOUT_DEFAULT
                totrew = [0] * args.players

                while True:
                    # Evaluate the current genome



                    if t % 10 == 0:

                        if verbosity > 1:
                            infostr = ''
                            if info:
                                infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
                            print(('t=%i' % t) + infostr)

                        if info is not None:
                            inputs = get_neat_inputs(ob, info, mario_config['NEAT'])
                        #outputs = net.activate(inputs)

                        env.render()


                    ob, rew, done, info = env.step(inputs)

                    t += 1

                    rew = [rew]
                    for i, r in enumerate(rew):
                        totrew[i] += r
                        if r > 0:
                            timeout = TIMEOUT_DEFAULT
                        else:
                            timeout -= 1
                            if timeout < 0:
                                done = True
                        if verbosity > 1:
                            if r > 0:
                                print('t=%i p=%i got reward: %g, current reward: %g' % (t, i, r, totrew[i]))
                            if r < 0:
                                print('t=%i p=%i got penalty: %g, current reward: %g' % (t, i, r, totrew[i]))
                    if done:
                        env.render()
                        print("GenomeId:%s time:%i fitness:%d" % (genome_id, t, totrew[0]))
                        genome.fitness = totrew[0]
                        break

            except KeyboardInterrupt:
                exit(0)

        # All genomes in the current population have been evaluated, get the best genome and move to the next generation
        p.next_generation()
        print("Best of gen: ", p.generation-1, "\nFitness: {!s}".format(p.best_genome.fitness))


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    main(config_path)
