import colorsys
import pickle
import argparse
import cv2
import os
import os.path

import numpy as np

from utilities.util import get_project_path


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_to", type=str, default="test_folder")
    parser.add_argument("--in_path", type=str, default=None)

    parser_args = parser.parse_args()
    return parser_args


def minmaxscaler(data):
    min = np.min(data)
    max = np.max(data)
    return (data - min) / (max - min)


def display_image(title: str, image: np.array):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def colorize_image(image: np.array, hue: float):
    scale = 10
    image = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale))
    # image = np.transpose(image, (1, 0))
    image_shape = image.shape
    new_image = np.zeros((image_shape[0], image_shape[1], 3)).astype(np.uint8)
    image = minmaxscaler(image)
    for i in range(len(image)):
        for j in range(len(image[i])):
            p = image[i][j]

            h = hue
            l = 0.3
            s = p * 1.5
            if s > 1:
                s = 1
            v = 1
            # rgb = colorsys.hls_to_rgb(h, l, s)
            rgb = colorsys.hsv_to_rgb(h, s, v)
            pr, pg, pb = [int(x * 255) for x in rgb]
            new_image[i, j, 0] = pb
            new_image[i, j, 1] = pg
            new_image[i, j, 2] = pr
    return new_image


def save_image(save_dir, name, image):
    cv2.imwrite(os.path.join(save_dir, name + ".png"), image)


def load_observation_map(path):
    observation_map = pickle.load(open(path, "rb"))
    return observation_map


def explode(data):
    size = np.array(data.shape) * 2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def get_observation_map():
    dir_path = os.path.join(get_project_path(), "output", "test_folder", "result_log")
    obs_rois = {}
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        observation_map = load_observation_map(file_path)
        observation_map = np.reshape(observation_map, (4, 5, 36, 18))
        observation_map = np.transpose(observation_map, (0, 2, 1, 3))
        observation_map = np.reshape(observation_map, (4, 36, 90))
        obs_roi = observation_map[1]
        obs_rois[file_name] = obs_roi
    pass


hue_map = {"unknown": 0.1, "free": 0.4, "occ": 0.6, "roi": 0.95}


def get_observation_map_first_layer(observation_maps):
    observation_maps = np.reshape(np.array(observation_maps), (-1, 4, 5, 36, 18))
    observation_maps = np.transpose(observation_maps, (0, 2, 1, 3, 4))

    max_roi_map = observation_maps[0][0]
    max_occ_map = observation_maps[0][1]
    max_free_map = observation_maps[0][2]
    max_unknown_map = observation_maps[0][3]

    max_roi_sum = 0
    max_roi_index = 0
    for i, observation_map in enumerate(observation_maps):
        # 5 x 36 x 18 (1).73 2
        observation_map_first_layer = observation_map[2].squeeze()

        unknown_map = observation_map_first_layer[0]
        free_map = observation_map_first_layer[1]
        occ_map = observation_map_first_layer[2]
        roi_map = observation_map_first_layer[3]

        roi_sum = np.sum(roi_map)

        print("roi sum : {}".format(roi_sum))

        if roi_sum > max_roi_sum:
            max_roi_sum = roi_sum
            max_roi_index = i
            max_roi_map = roi_map
            max_occ_map = occ_map
            max_free_map = free_map
            max_unknown_map = unknown_map

    roi_map = minmaxscaler(max_roi_map)
    occ_map = minmaxscaler(max_occ_map)
    free_map = minmaxscaler(max_free_map)
    unknown_map = minmaxscaler(max_unknown_map)
    print("max roi index:{}".format(max_roi_index))
    print("max roi sum:{}".format(max_roi_sum))

    next_roi_map = minmaxscaler(observation_maps[max_roi_index + 1][2].squeeze()[0])
    next_occ_map = minmaxscaler(observation_maps[max_roi_index + 1][2].squeeze()[1])
    next_free_map = observation_maps[max_roi_index + 1][2].squeeze()[2]
    next_unknown_map = observation_maps[max_roi_index + 1][2].squeeze()[3]
    result = {}
    result["unknown_map"] = colorize_image(unknown_map, hue_map["unknown"])
    result["free_map"] = colorize_image(free_map, hue_map["free"])
    result["occ_map"] = colorize_image(occ_map, hue_map["occ"])
    result["roi_map"] = colorize_image(roi_map, hue_map["roi"])

    result["unknown_map_next"] = colorize_image(next_unknown_map, hue_map["unknown"])
    result["free_map_next"] = colorize_image(next_free_map, hue_map["free"])
    result["occ_map_next"] = colorize_image(next_occ_map, hue_map["occ"])
    result["roi_map_next"] = colorize_image(next_roi_map, hue_map["roi"])

    return result


if __name__ == '__main__':
    result_dir = os.path.join(get_project_path(), "output", "observation_map", "result_log")
    observation_map_root_dir = os.path.join(result_dir, "observation_map")
    observation_map_paths = [os.path.join(observation_map_root_dir, name) for name in
                             os.listdir(observation_map_root_dir)]
    observation_maps = []
    for observation_map_path in observation_map_paths:
        step_count, reward, action, observation_map = load_observation_map(observation_map_path)
        observation_maps.append(observation_map)

    result = get_observation_map_first_layer(observation_maps)

    for key in result.keys():
        save_image(result_dir, key, result[key])
