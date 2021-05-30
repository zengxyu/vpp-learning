import numpy as np


def trim_zeros(arr):
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(arr))
    return arr[slices]


def paste_slices(tup):
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos + w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w - max(pos + w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)


def paste(wall, block, loc):
    if block.shape[0] + loc[0] >= wall.shape[0] or block.shape[1] + loc[1] >= wall.shape[1] or block.shape[2] + loc[
        2] >= wall.shape[2]:
        return None
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] = block[block_slices]
    return wall


#
# def paste(wall, block, position):
#     x, y, z = position
#     wall[x:x + block.shape[0], y:y + block.shape[1], z:z + block.shape[2]] = block
#     return wall


if __name__ == '__main__':
    # arr = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 1, 2, 3, 0, 0, 0, 0],
    #                 [0, 0, 0, 4, 5, 6, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    #
    arr2 = np.array([[[0, 1, 2, 3], [0, 1, 2, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
                     [[0, 1, 2, 3], [0, 1, 2, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
                     [[0, 1, 2, 3], [0, 1, 2, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
                     [[0, 1, 2, 3], [0, 1, 2, 0], [0, 1, 0, 0], [1, 0, 0, 0]]])

    wall = np.zeros((10, 10, 10))
    result = paste(wall, arr2, (7, 7, 7))
    print(result)
