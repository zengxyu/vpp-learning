roslaunch ur_with_cam_gazebo ur_with_cam.launch base:=retractable world_name:=world19 gui:=false
rosrun vpp_learning_ros vpp_env_server_node _evaluate_results:=true
python3 run_ros_discrete.py rl_policy --num_episodes=20 --in_folder=world19_random5 --in_model_index=1800 --out_folder=evaluate_world19_random5 --handle_simulation=0
