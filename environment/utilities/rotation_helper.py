# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/17/22 11:13 PM 
    @Description    :
        
===========================================
"""
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation


def rotation_to_py_quaternion(rot: Rotation):
    """
    convert rotation:Rotation to quaternion: Quaternion
    """
    rot_q = rot.as_quat()
    rot_q = Quaternion(rot_q[3], rot_q[0], rot_q[1], rot_q[2])
    return rot_q


def py_quaternion_to_rotation(quaternion: Quaternion):
    """
    convert quaternion: Quaternion to rotation: Rotation
    """
    d_q = quaternion.q
    rot = Rotation.from_quat(np.array([d_q[1], d_q[2], d_q[3], d_q[0]]))
    return rot


def get_rotation_between_rotations(rot1: Rotation, rot2: Rotation):
    """
    get rotation between two rotations, rot2 = rot * rot1, given rot1 and rot2, get rot?
    """
    rot1_q = rotation_to_py_quaternion(rot1)
    rot2_q = rotation_to_py_quaternion(rot2)
    rot_q = rot2_q * rot1_q.inverse
    rot = py_quaternion_to_rotation(rot_q)
    return rot
