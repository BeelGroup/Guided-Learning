import sys
import numpy as np


def stitch_tiles(tiles, tile_width, tile_height):
    ''' Rebuilds the given tile list (in col major order) to an image. Mostly for testing.
            Returns: 3D numpy array'''
    # print(tiles.shape)
    ret = np.empty((tiles.shape[1] * tile_width, 0, tiles.shape[4]), dtype=np.uint8)
    for col in tiles:
        tmp = np.empty((0, tile_height, tiles.shape[4]), dtype=np.uint8)
        for row in col:
            tmp = np.concatenate((tmp, row), axis=0)
        ret = np.concatenate((ret, tmp), axis=1)
    return ret


def pad(array, reference_shape, offset, dtype):
    """
    array: Array to be padded
    reference: A tuple of the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.full(reference_shape, 255, dtype=dtype)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offset[dim], offset[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    print("[pad] array.shape: {}".format(array.shape))
    print("[pad] result.shape: {}".format(result.shape))
    return result


def get_raw_enemy_pos(info):
    ''' Returns: A list of the center coordinate of the enemies as (x, y)
        (-1,-1) if enemy not drawn'''
    enemy_pos = []
    for i in range(1, 6):
        enemy_drawn_str = "enemy_{}_drawn".format(i)
        if info[enemy_drawn_str]:
            enemy_hitbox_str = "enemy_{}_hitbox_".format(i)
            enemy_pos.append((info[enemy_hitbox_str +"x2"] - info[enemy_hitbox_str +"x1"],
                              info[enemy_hitbox_str +"y2"] - info[enemy_hitbox_str +"y1"]))
        else:
            enemy_pos.append((-1 ,-1))
    return enemy_pos

def get_raw_player_pos(info):
    ''' Returns: The center coordinate of the player as (x, y)'''
    hitbox_width = info['player_hitbox_x2'] - info['player_hitbox_x1']
    hitbox_height = info['player_hitbox_y2'] - info['player_hitbox_y1']
    hitbox_center = (info['player_hitbox_x1'] + (hitbox_width /2), info['player_hitbox_y1'] + (hitbox_height /2))
    return hitbox_center
