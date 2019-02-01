
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
    ''' Takes a numpy array of tiles and returns all tiles of distance d from center (x,y) of the tiles array
        The output array is padded as necessary '''

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

    return (p_col, p_row)
