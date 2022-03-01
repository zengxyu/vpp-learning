python3 evaluate_p3d.py "circular_policy" --out_folder=evaluate_circular_policy/320x320x150_2plants --room_size 320 320 150 --plant_num_choices 2 --num_episodes=30 --max_steps=400 --random_plant_number=True --randomize_sensor_position=0;
python3 evaluate_p3d.py "circular_policy" --out_folder=evaluate_circular_policy/400x400x150_2plants --room_size 400 400 150 --plant_num_choices 2 --num_episodes=30 --max_steps=400 --random_plant_number=True --randomize_sensor_position=0;
python3 evaluate_p3d.py "circular_policy" --out_folder=evaluate_circular_policy/480x480x150_2plants --room_size 480 480 150 --plant_num_choices 2 --num_episodes=30 --max_steps=400 --random_plant_number=True --randomize_sensor_position=0;
python3 evaluate_p3d.py "circular_policy" --out_folder=evaluate_circular_policy/320x480x150_2plants --room_size 320 480 150 --plant_num_choices 2 --num_episodes=30 --max_steps=400 --random_plant_number=True --randomize_sensor_position=0;

