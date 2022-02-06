from rl_agents.action_space.action_space import ActionMoRo10 as Action
import field_env_3d_helper
import numpy as np

arr = field_env_3d_helper.generate_test_vector_array()
print(arr)
print(type(arr))
print(arr.dtype)

field = Field(Action, shape=(256, 256, 256), sensor_range=50, hfov=90.0, vfov=60.0, scale=0.05, max_steps=300, init_file='VG07_6.binvox', headless=True)
field.reset()
spherical_coordinate_map = field.generate_spherical_coordinate_map(field.robot_pos)
print(spherical_coordinate_map)
print(np.histogram(spherical_coordinate_map))
