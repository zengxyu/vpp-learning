import roslaunch
import time

# Prepare launch of simulation

uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)

launch_file = roslaunch.rlutil.resolve_launch_arguments(['vpp_learning_ros', 'ur_with_cam.launch'])
launch_args = ['world_name:=world19']

launch_files = [(launch_file[0], launch_args)]

parent = roslaunch.parent.ROSLaunchParent(uuid, launch_files)

parent.start()

for i in range(10):
    print(i)
    parent.spin_once()
    time.sleep(1)

print('Shutting down')
parent.shutdown()
print('Shutdown complete')

parent = roslaunch.parent.ROSLaunchParent(uuid, launch_files)
parent.start()

for i in range(10):
    print(i)
    parent.spin_once()
    time.sleep(1)

print('Shutting down')
parent.shutdown()
print('Shutdown complete')
