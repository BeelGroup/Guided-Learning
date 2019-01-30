#!/usr/bin/env python

import argparse, os, sys, time, math
import retro

import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import minmax_scale

from skimage import color
from skimage.measure import block_reduce
from skimage.segmentation import felzenszwalb, slic

import neat
from neat.six_util import iteritems
import visualize

from curses import wrapper


parser = argparse.ArgumentParser()
parser.add_argument('--game', default='SuperMarioBros-Nes', help='the name or path for the game to run')
parser.add_argument('--state', default='Level1-1', help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario.json', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
args = parser.parse_args()

env = retro.make(args.game, args.state or retro.State.DEFAULT, scenario=args.scenario, record=args.record, players=args.players)
verbosity = args.verbose - args.quiet

TIMEOUT_DEFAULT = 75


def get_inputs(frame, info):
    ''' Uses the given frame to compute an output for each 16x16 block of pixels.
        Extracts player position and enemy positions (-1 if unavailable) from info.
            Returns: A list of inputs to be fed to the NEAT network '''
    player_pos = get_player_pos(info)
    print("Player Pos: {}".format(player_pos))
    tiles = get_tiles(frame, player_pos, 2, info['xscrollLo'])
    ret = []
    for tile in tiles:
        # convert tile to greyscale, normalize and compute the mean
        ret.append(np.mean(minmax_scale(color.rgb2gray(tile))))



def classify_tiles(tiles):
    # TODO
    pass


def align_frame(frame, x_scroll_lo):
    ''' Aligns the current frame to 16x16 tiles based on x_scroll_lo
            Returns: The aligned frame'''
    left_diff = 16 - ((x_scroll_lo + 8) % 16)
    right_diff = -1 * (16 - left_diff)  # can be 0 if x_scroll_lo == 8
    # adjust the image (remove 8 pixels from the bottom, left and right)
    if right_diff == 0:
        aligned = frame[8:-8, left_diff:]
    else:
        aligned = frame[8:-8, left_diff:right_diff]
    return aligned


def get_surrounding_tiles(tiles, center, d=1):
    ''' Takes a numpy array of tiles and returns all points of distance d from center (x,y) of the tiles array'''
    print("Center Tile: {}".format(center))
    t = tiles[center[0]+1-d:center[0]+1+d+1, center[1]-1-d:center[1]-1+d+1]
    return t


def stitch_tiles(tiles):
    ''' Rebuilds the given tile list (in row major order) to an image
            Returns: 3D numpy array'''
    ret = np.empty((tiles.shape[0] * 16, 0, tiles.shape[4]), dtype=np.uint8)
    for r in tiles:
        tmp = np.empty((0, 16, tiles.shape[4]), dtype=np.uint8)
        for c in r:
            tmp = np.concatenate((tmp, c), axis=0)
        ret = np.concatenate((ret, tmp), axis=1)
    return ret


def get_tiles_from_frame(frame, tile_width, tile_height):
    ''' Tiles the given frame into images of the given size
            Returns: A numpy array of tiles in row major order'''

    tiles = np.empty((int(frame.shape[0]/16), int(frame.shape[1]/16), 16, 16, 3), dtype=np.uint8)

    # TODO: Convert to ROW MAJOR (swap tiles[col_count, row_count]) but the player positioning breaks
    col_count = 0
    for y in range(0, frame.shape[0], tile_height):
        row_count = 0
        for x in range(0, frame.shape[1], tile_width):
            t = frame[x:x + 16, y:y + 16] # TODO: verify x and y are correct here
            if t.shape == (16, 16, 3):
                tiles[col_count, row_count, :, :, :] = t
                row_count += 1
        col_count += 1
    return tiles


def get_tile_index_of_player_pos(tiles, player_pos):
    # reverse coordinates for (0,0) on bottom left to (0,0) on top left
    # player_pos is in the form (y, x)
    p = (player_pos[1], (tiles.shape[0]*16)-player_pos[0])
    p_row = None
    p_col = None
    for i, row in enumerate(tiles):
        if p[1] > i*16 and p[1] < (i+1)*16:
            p_row = i
        else:
            continue
        for j, col in enumerate(row):
            if p[0] > j*16 and p[0] < (j+1)*16:
                p_col = j
    assert(p_row is not None and p_col is not None)
    return (p_col, p_row)




def get_radius_tiles(frame, player_pos, radius):
    tiles = get_tiles_from_frame(frame, 16, 16)
    player_tile_pos = get_tile_index_of_player_pos(tiles, player_pos)


    closest = get_surrounding_tiles(tiles, player_tile_pos, 1)
    '''
    for row in closest:
        for col in row:
            plt.imshow(col)
            plt.show()
    '''

    s = stitch_tiles(closest)
    plt.imshow(s)
    plt.show()


def get_tiles(frame, player_pos, radius, x_scroll_lo, display_tiles=False):
    ''' Gets the tiles around the given center and inside radius.
        The tiles are aligned using x_scroll_lo.
            Returns: A list of 16x16 tiles'''

    aligned = align_frame(frame, x_scroll_lo)

    tiles = get_radius_tiles(aligned, player_pos, radius)
    for row in tiles:
        for col in row:
            plt.imshow(col)
            plt.show()
    tiles = [j for sub in tiles for j in sub] # flatten to 1D

    if display_tiles:
        print("Generating tile graph of the current frame..")

        s = int(math.ceil(math.sqrt(len(tiles))))
        fig2, ax2 = plt.subplots(nrows=s - 1, ncols=s)
        for i, row in enumerate(ax2):
            for j, col in enumerate(row):
                try:
                    col.imshow(tiles[(s * i) + j])
                except IndexError:
                    break
                col.axes.get_yaxis().set_visible(False)
                col.axes.get_xaxis().set_visible(False)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        major_ticks = np.arange(0, 256, 16)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.imshow(aligned)
        plt.grid(True)

        print("Done.")

        plt.show()

    return tiles


def get_player_pos(info):
    ''' Returns: The center coordinate of the player as (x, y)'''
    return (info['player_hitbox_x2']-info['player_hitbox_x1'], info['player_hitbox_y2']- info['player_hitbox_y1'])


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
                            inputs = get_inputs(ob, info)
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
