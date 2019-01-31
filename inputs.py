
import sys, math
import numpy as np
from matplotlib import pyplot as plt

from skimage import color

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

    player_pos = get_player_pos(info)
    inputs.append(player_pos[0])
    inputs.append(player_pos[1])

    enemy_pos = get_enemy_pos(info)
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
        print("[get_neat_inputs] Player Pos: {}".format(player_pos))
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
    ''' Takes a numpy array of tiles and returns all tiles of distance d from center (x,y) of the tiles array'''
    # bound the indices (an index of -1 will return None)
    l1_index = center[0] + 1 - radius
    if l1_index < 0: l1_index = 0
    r1_index = center[0] + 1 + radius + 1
    if r1_index < 0: r1_index = 0
    l2_index = center[1] - 1 - radius
    if l2_index < 0: l2_index = 0
    r2_index = center[1] - 1 + radius + 1
    if r2_index < 0: r2_index = 0
    if r2_index > 13: r2_index = 13 # There's hidden data at 14 (bottom of the screen) that is just noise and does not normally display
    t = tiles[l1_index:r1_index, l2_index:r2_index]
    if __debug__:
        print("[get_surrounding_tiles] Center Tile: {}".format(center))
        print("[get_surrounding_tiles] indexing: {}:{}, {}:{}".format(l1_index, r1_index, l2_index, r2_index))
    return t


def get_tile_index_of_player(tiles, player_pos, tile_width, tile_height):
    # reverse coordinates for (0,0) on bottom left to (0,0) on top left. player_pos is in the form (y, x)
    p = (player_pos[1], (tiles.shape[0] * tile_width) - player_pos[0])
    p_row = None
    p_col = None
    for i, row in enumerate(tiles):
        if p[1] > i * tile_width and p[1] < (i + 1) * tile_width:
            p_row = i
        else:
            continue
        for j, col in enumerate(row):
            if p[0] > j * tile_height and p[0] < (j + 1) * tile_height:
                p_col = j
    assert (p_row is not None and p_col is not None)
    return (p_col, p_row)


def stitch_tiles(tiles, tile_width, tile_height):
    ''' Rebuilds the given tile list (in col major order) to an image. Mostly for testing.
            Returns: 3D numpy array'''
    #print(tiles.shape)
    ret = np.empty((tiles.shape[1] * tile_width, 0, tiles.shape[4]), dtype=np.uint8)
    for col in tiles:
        tmp = np.empty((0, tile_height, tiles.shape[4]), dtype=np.uint8)
        for row in col:
            tmp = np.concatenate((tmp, row), axis=0)
        ret = np.concatenate((ret, tmp), axis=1)
    return ret


def get_enemy_pos(info):
    enemy_pos = []
    for i in range(1, 6):
        enemy_drawn_str = "enemy_{}_drawn".format(i)
        if info[enemy_drawn_str]:
            enemy_hitbox_str = "enemy_{}_hitbox_".format(i)
            enemy_pos.append((info[enemy_hitbox_str+"x2"] - info[enemy_hitbox_str+"x1"],
                              info[enemy_hitbox_str+"y2"] - info[enemy_hitbox_str+"y1"]))
        else:
            enemy_pos.append((-1,-1))
    return enemy_pos

def get_player_pos(info):
    ''' Returns: The center coordinate of the player as (x, y)'''
    return (info['player_hitbox_x2'] - info['player_hitbox_x1'], info['player_hitbox_y2'] - info['player_hitbox_y1'])
