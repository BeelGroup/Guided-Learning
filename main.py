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
args = parser.parse_args()


def main(config_file):
    env = retro.make(args.game, args.state or retro.State.DEFAULT, scenario=args.scenario, record=args.record,
                     players=args.players)

    if args.load != '':
        mario = load_state("saves/"+args.load)
        mario.set_env(env)
    else:
        mario = Mario(env, config_file)

    mario.run()


if __name__ == "__main__":
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.config')
    main(config_path)
