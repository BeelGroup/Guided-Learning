#!/usr/bin/env python

import argparse, os, sys
import retro

import numpy as np
from matplotlib import pyplot as plt

from skimage import color
from skimage.measure import block_reduce
from skimage.segmentation import felzenszwalb, slic

import neat
from neat.six_util import iteritems
import visualize

parser = argparse.ArgumentParser()
parser.add_argument('--game', default='SuperMarioBros-Nes', help='the name or path for the game to run')
parser.add_argument('--state', default='Level1-1', help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='neat_scenario.json', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
args = parser.parse_args()

env = retro.make(args.game, args.state or retro.State.DEFAULT, scenario=args.scenario, record=args.record, players=args.players)
verbosity = args.verbose - args.quiet

TIMEOUT_DEFAULT = 75


def get_inputs(ob):
    greyed_ob = color.rgb2gray(color.yiq2rgb(ob))
    # perform downsampling on the array
    reduced_greyed_ob = block_reduce(greyed_ob, (4, 4))
    return reduced_greyed_ob.flatten()


def get_tiles(frame, x_scroll_lo):
    # returns the 16x16 tiles from the given frame

    diff = (x_scroll_lo + 8) % 16

    # adjust the image (remove 8 pixels from the bottom, left and right)
    cropped = frame[:-8, diff:-diff]

    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.imshow(frame)
    fig.add_subplot(1, 2, 2)
    plt.imshow(cropped)

    plt.show()

    return [cropped[x:x+16, y:y+16] for x in range(0, cropped.shape[0], 16) for y in range(0, cropped.shape[1], 16)]


def segment_frame(frame):
    '''
    grey_frame = color.rgb2gray(frame)
    edges = canny(grey_frame)
    fill = ndi.binary_fill_holes(edges)
    label_objects, nb_labels = ndi.label(fill)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > 20
    mask_sizes[0] = 0
    cleaned = mask_sizes[label_objects]
    '''

    frame = color.rgb2hsv(frame)

    fig = plt.figure()

    fig.add_subplot(2, 2, 1)
    plt.imshow(frame, interpolation='nearest')
    fig.add_subplot(2, 2, 2)
    plt.imshow(slic(frame, n_segments=30, compactness=0.00001, sigma=1))
    fig.add_subplot(2, 2, 3)
    plt.imshow(slic(frame, n_segments=20, compactness=10, sigma=1))
    fig.add_subplot(2, 2, 4)
    plt.imshow(felzenszwalb(frame, sigma=1.5), interpolation='nearest', cmap="gray")   # USE THIS

    plt.show()

    sys.exit(0)


def main(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    print("Generating population..")
    p = neat.Population(config)
    print("Population Generated.")

    info = None
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
                        #if info:
                        #    getTiles(ob, info['xscrollLo'])

                        #segment_frame(ob)

                        inputs = get_inputs(ob)

                        outputs = net.activate(inputs)

                        env.render()

                    ob, rew, done, info = env.step(outputs)

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
