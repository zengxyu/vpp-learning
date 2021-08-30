import pickle

import numpy as np
import cv2
import torch
import os

save_path = "/home/zeng/workspace/vpp-learning/results_paper/visualize_observation_map"


def save_image(name, image):
    image_path = os.path.join(save_path, name)
    cv2.imwrite(image_path, image * 256)


def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min) / (max - min)


def con_frame(frame):
    con_frames = frame[0]
    for i in range(1, 15, 1):
        br = np.ones((frame[i].shape[0], 2))
        con_frames = np.hstack([con_frames, br, frame[i]])

    return con_frames


def con_frame2(frame):
    results = None
    for i in range(5):
        a = [frame[0 + i]]
        a.append(frame[5 + i])
        a.append(frame[10 + i])
        a = np.array(a)
        a = a.transpose((1, 2, 0))
        if results is None:
            results = [a]
        else:
            results.append(a)
        save_image("observation_map_{}.png".format(i), a)
    results = np.hstack(results)
    return results


def display_image(image, title):
    scale = 10
    image = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale))
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


path = "/home/zeng/workspace/vpp-learning/output/out_p3d_temporal_pose_random_108_control2/experience/buffer.obj"
exp_obj = open(path, 'rb')
memory = pickle.load(exp_obj)
memory.seq_len = 10
tree_idx, minibatch, ISWeights = memory.sample(is_vpp=True)
states, actions, rewards, next_states, dones = minibatch

# 最大奖励的reward
index = np.argmax(rewards)
# index = 5
print(rewards[index])
frames, poses = states
frames = minmaxscaler(frames).numpy()
frame = frames[index]

frames_next, poses_next = next_states
frames_next = minmaxscaler(frames_next).numpy()
frame_next = frames_next[index]

resize_frame = con_frame(frame)
resize_frame_next = con_frame(frame_next)
vertical_con_frame = np.vstack([resize_frame, resize_frame_next])
cf2 = con_frame2(frame)
display_image(cf2, "color")
# display_image(vertical_con_frame, "resize_frame")
# display_image(resize_frame_next, "resize_frame_next")

# frame = cv2.resize(frame[0], dsize=(frame[0].shape[1] * 5, frame[0].shape[0] * 5))
