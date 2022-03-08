roslaunch ur_with_cam_gazebo ur_with_cam.launch base:=retractable world_name:=world19 gui:=false
rosrun vpp_learning_ros vpp_env_server_node _evaluate_results:=true
python3 run_ros_discrete.py rl_policy --num_episodes=20 --in_folder=world19_random4_maxlen150_change_visit_map --in_model_index=615 --out_folder=evaluate_world19_random --epsilon_greedy --epsilon=0.15
