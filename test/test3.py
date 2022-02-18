#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
===========================================
    @Project : vpp-learning 
    @Author  : Xiangyu Zeng
    @Date    : 2/16/22 5:51 PM 
    @Description    :
        
===========================================
"""
import numpy as np
from scipy.spatial.transform import Rotation

a = Rotation.from_euler('zyx', [15, 0, 0], degrees=True)
b = Rotation.from_euler('zyx', [30, 12, 0], degrees=True)
c = Rotation.from_euler('zyx', [45, 12, 0], degrees=True)
d = c * b.inv()

print(a.as_euler('zyx', degrees=True))
print(d.as_euler('zyx', degrees=True))
