
import math
import numpy as np
from matplotlib import pyplot as plt


def get_inputs(frame, info):
    '''Uses the given frame to compute an output for each 16x16 block of pixels.
        Extracts player position and enemy positions (-1 if unavailable) from info.
            Returns: A list of inputs to be fed to the NEAT network '''
    player_pos = get_player_pos(info)
    print("Player Pos: {}".format(player_pos))

    # Convert frame to greyscale?
    tiles = get_tiles(frame, 16, 16, info['xscrollLo'], player_pos=player_pos, radius=1)
    for t in tiles:
        plt.imshow(t)
        plt.show()


def get_tiles(frame, tile_width, tile_height, x_scroll_lo, player_pos=None, radius=None, display_tiles=False):
    ''' Tiles the given frame. Optionally gets the tiles surrounding the given player_pos within radius
            Returns: A list of 16x16 tiles'''

    # align the frame correctly
    aligned = align_frame(frame, x_scroll_lo)

    # get the 16x16 tiles from the aligned frame
    tiles = tile_frame(aligned, tile_width, tile_height)

    if player_pos is not None and radius is not None:
        # get the index of the tile where the player is
        player_tile_pos = get_tile_index_of_player(tiles, player_pos, tile_width, tile_height)
        tiles = get_surrounding_tiles(tiles, player_tile_pos, radius)

    # flatten to 3D
    tiles = [j for sub in tiles for j in sub]

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

    # TODO: Convert to ROW MAJOR (swap tiles[col_count, row_count]) but the player positioning breaks
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
    # TODO: fix for large radius
    ''' Takes a numpy array of tiles and returns all points of distance d from center (x,y) of the tiles array'''
    print("[get_surrounding_tiles] Center Tile: {}".format(center))
    t = tiles[center[0] + 1 - radius:center[0] + 1 + radius + 1, center[1] - 1 - radius:center[1] - 1 + radius + 1]
    #print(t)
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
    # TODO: Convert to ROW MAJOR
    ''' Rebuilds the given tile list (in col major order) to an image. Mostly for testing.
            Returns: 3D numpy array'''
    ret = np.empty((tiles.shape[0] * tile_width, 0, tiles.shape[4]), dtype=np.uint8)
    for r in tiles:
        tmp = np.empty((0, tile_height, tiles.shape[4]), dtype=np.uint8)
        for c in r:
            tmp = np.concatenate((tmp, c), axis=0)
        ret = np.concatenate((ret, tmp), axis=1)
    return ret


def get_player_pos(info):
    ''' Returns: The center coordinate of the player as (x, y)'''
    return (info['player_hitbox_x2'] - info['player_hitbox_x1'], info['player_hitbox_y2'] - info['player_hitbox_y1'])
