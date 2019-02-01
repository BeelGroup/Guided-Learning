
import sys
import numpy as np
from matplotlib import pyplot as plt

from utils import *

import pyglet

def get_keyboard_input(obs, win):
    key_handler = pyglet.window.key.KeyStateHandler()

    joysticks = pyglet.input.get_joysticks()
    if len(joysticks) > 0:
        joystick = joysticks[0]
        joystick.open()
    else:
        joystick = None

    win.push_handlers(key_handler)

    key_previous_states = {}
    button_previous_states = {}

    pyglet.app.platform_event_loop.start()

    while True:
        win.dispatch_events()

        win.clear()

        keys_clicked = set()
        keys_pressed = set()

        for key_code, pressed in key_handler.items():
            if pressed:
                keys_pressed.add(key_code)

            if not key_previous_states.get(key_code, False) and pressed:
                keys_clicked.add(key_code)
            key_previous_states[key_code] = pressed

        buttons_clicked = set()
        buttons_pressed = set()
        if joystick is not None:
            for button_code, pressed in enumerate(joystick.buttons):
                if pressed:
                    buttons_pressed.add(button_code)

                if not button_previous_states.get(button_code, False) and pressed:
                    buttons_clicked.add(button_code)
                button_previous_states[button_code] = pressed

        print(keys_pressed)

    sys.exit(0)





def get_neat_inputs(frame, info, config):
    '''Uses the given frame to compute an output for each 16x16 block of pixels.
        Extracts player position and enemy positions (-1 if unavailable) from info.
            Returns: A list of inputs to be fed to the NEAT network '''

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

    if config.getboolean('inputs_greyscale'):
        frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
        frame.resize((frame.shape[0], frame.shape[1], 1))

    tiles = get_tiles(frame, 16, 16, info['xscrollLo'], player_pos=player_pos, radius=3)

    # Get an array of average values per tile (this may have 3 values for RGB or 1 for greyscale)
    tile_avg = np.mean(tiles, axis=3, dtype=np.uint16)

    if config.getboolean('inputs_greyscale'):
        # the greyscale value is in all 3 positions so just get the first
        tile_inputs = tile_avg[:, :, :, 0:1].flatten().tolist()
    else:
        tile_inputs = tile_avg[:, :, :, :].flatten().tolist()

    inputs = inputs + tile_inputs

    if __debug__:
        print("[get_neat_inputs] Raw Player Pos: {}".format(player_pos))
        print("[get_neat_inputs] Enemy Pos: {}".format(enemy_pos))
        print("[get_neat_inputs] Input Length: {}".format(len(inputs)))
        #print("[get_neat_inputs] Inputs: {}".format(inputs))
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


def get_tiles(frame, tile_width, tile_height, x_scroll_lo, player_pos=None, radius=None, display_tiles=False):
    ''' Tiles the given frame. Optionally gets the tiles surrounding the given player_pos within radius
            Returns: A numpy array of 16x16 tiles'''

    # align the frame correctly
    aligned = align_frame(frame, x_scroll_lo)

    # get the 16x16 tiles from the aligned frame
    tiles = tile_frame(aligned, tile_width, tile_height)

    # remove noise from the 13th row
    tiles[:,13,:,:,:] = 255


    if player_pos is not None and radius is not None:
        # get the index of the tile where the player is
        player_tile_pos = get_tile_index_of_player(tiles, player_pos, tile_width, tile_height)

        print("[get_tiles] tiles.shape: {}".format(tiles.shape))
        tiles = get_surrounding_tiles(tiles, player_tile_pos, radius)

    return tiles


def align_frame(frame, x_scroll_lo):
    # TODO: Generalize
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


def tile_frame(frame, tile_width, tile_height):
    ''' Tiles the given frame into images of the given size
            Returns: A numpy array of tiles in row major order'''

    tiles = np.empty((int(frame.shape[0] / tile_width), int(frame.shape[1] / tile_height), tile_width, tile_height, 3), dtype=np.uint8)

    col_count = 0
    for y in range(0, frame.shape[0], tile_height):
        row_count = 0
        for x in range(0, frame.shape[1], tile_width):
            t = frame[x:x + tile_width, y:y + tile_height]  # TODO: verify x and y are correct here
            if t.shape == (tile_width, tile_height, frame.shape[2]):
                tiles[col_count, row_count, :, :, :] = t
                row_count += 1
        col_count += 1
    return tiles


def get_surrounding_tiles(tiles, center, radius=1):
    ''' Takes a numpy array of tiles and returns all tiles of distance d from center (x,y) of the tiles array'''
    # TODO: THIS MAKES THE INPUT SIZE A VARIABLE LENGTH, MUST PAD

    left_col_index = center[0] - 1 - radius
    right_col_index = center[0] + radius
    top_row_index = center[1] - 1 - radius
    bottom_row_index = center[1] + radius

    # 14 rows, 13 columns of tiles
    add_left_cols = abs(left_col_index) if left_col_index < 0 else 0
    add_right_cols = right_col_index - 13 if right_col_index > 13 else 0
    add_top_rows = abs(top_row_index) if top_row_index < 0 else 0
    add_bottom_rows = bottom_row_index - 14 if bottom_row_index > 14 else 0  # -1 due to noise row at bottom

    # bound the indices
    if left_col_index < 0: left_col_index = 0
    if top_row_index < 0: top_row_index = 0

    tiles = tiles[left_col_index:right_col_index, top_row_index:bottom_row_index]

    tiles_pad = pad(tiles, (tiles.shape[0] + add_left_cols+add_right_cols, tiles.shape[1]+add_bottom_rows+add_top_rows, tiles.shape[2], tiles.shape[3], tiles.shape[4]),
                    (add_left_cols, add_top_rows, 0, 0, 0), dtype=np.uint8)

    if __debug__:
        print("[get_surrounding_tiles] tiles.shape: {}".format(tiles.shape))
        print("[get_surrounding_tiles] tiles_pad.shape: {}".format(tiles_pad.shape))
        print("[get_surrounding_tiles] Center Tile: {}".format(center))
        print("[get_surrounding_tiles] indexing: {}:{}, {}:{}".format(left_col_index, right_col_index,
                                                                      top_row_index, bottom_row_index))
        print(
            "[get_surrounding_tiles] adjust:\nAdd {} columns to the left\nAdd {} columns to the right\nAdd {} rows to the top\nAdd {} rows to the bottom".format(
                add_left_cols, add_right_cols, add_top_rows, add_bottom_rows))

    return tiles_pad


def get_tile_index_of_player(tiles, player_pos, tile_width, tile_height):
    p = player_pos

    p_row = None
    p_col = None

    # bound the player position
    if p[0] < 0:
        p = (0, p[1])
    if p[1] < 0:
        p = (p[0], 0)
    if p[1] > 208:
        p = (p[0], 208)

    for i, row in enumerate(tiles):
        if p[1] > i * tile_height and p[1] <= (i + 1) * tile_height:
            p_row = i
        else:
            continue
        for j, col in enumerate(row):
            if p[0] > j * tile_width and p[0] <= (j + 1) * tile_width:
                p_col = j
    assert (p_col is not None and p_row is not None)

    print("[get_tile_index_of_player] tile_index: {}".format((p_col, p_row)))

    return (p_col, p_row)
