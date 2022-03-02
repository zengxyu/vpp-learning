# VPP learning

Contains environments used for viewpoint planning learning

# Compile p3d environment

```compile_pybind11_only.sh```

# Compile p3d environment with GUI

```compile_p3d_modules.sh```

# Train

``` python3 run_p3d_discrete.py rl_policy --out_folder=test_folder --train```

# Test, you need to specify your in_folder, out_folder,

# in_model_index(you can look for the model index in your in_folder/model directory)

```python3 run_p3d_discrete.py  rl_policy --in_folder=in_folder --out_folder=out_folder --in_model_index=2000```

The reinforcement learning lib from https://github.com/pfnet/pfrl.git
