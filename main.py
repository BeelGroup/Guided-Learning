#!/usr/bin/env python

import argparse, os, pickle
import retro

from utils import *
from mario import Mario


parser = argparse.ArgumentParser()
parser.add_argument('--game', default='SuperMarioBros-Nes', help='the name or path for the game to run')
parser.add_argument('--state', default='Level1-1', help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario.json', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
parser.add_argument('--load', '-l', default='', help='the mario state filename to load')
parser.add_argument('--limit', '-f', type=int, default=-1, help='limit the fps (default -1 [no limit])')
parser.add_argument('--disable_gen_stats', '-d', default=False, action='store_true', help='disable statistics and backup')
parser.add_argument('--name', default='1', help='the name given to the run')
args = parser.parse_args()


def main(config_file):
    env = retro.make(args.game, args.state or retro.State.DEFAULT, scenario=args.scenario, record=args.record,
                     players=args.players)

    if args.load != '':
        print("Loading previous Mario state..")
        mario = load_state("saves/run_{}/{}.bkup".format(args.name, args.load))
        mario.set_env(env)
        if args.name == '1':
            print("Using previous run name: {}".format(mario.get_run_name()))
        print("Loaded.")
    else:
        mario = Mario(env, config_file)
        mario.run_name = args.name

    dirs = ['eval/run_{}'.format(args.name), 'saves/run_{}'.format(args.name)]
    for directory in dirs:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    mario.run(fps=int(args.limit), gen_stats=not args.disable_gen_stats)


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.config')
    main(config_path)
