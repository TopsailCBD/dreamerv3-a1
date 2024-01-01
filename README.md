# DreamerV3-A1

### Install 

### The first stage, using env isaacgym.

1. install miniconda:
    - `bash Miniconda3-4.7.12-Linux-x86_64.sh`
2. Create a new python virtual env with python 3.8 (3.8 recommended). i.e. with conda:
    - `conda create -n isaacgym python==3.8`
    - `conda activate isaacgym`
3. Install pytorch 1.10 with cuda-11.3:
    - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Install Isaac Gym, copy the isaacgym folder to your own path, i.e., ~/, and:
   - `cd isaacgym/python && pip install -e .`
4. Install other dependence
   -  `pip install -r requirements` 
5. Troubleshooting
   - `sudo apt-get update`
   - `sudo apt-get install build-essential --fix-missing`

   - `pip install setuptools==59.5.0`

### Troubleshooting

1. `AttributeError: module 'distutils' has no attribute 'version'`
    - 'Troubleshooting' in last section
    - `pip install setuptools==59.5.0`
    - for some reason setuptools is reinstalled to higher version even after cloning an env.
2. `ModuleNotFoundError: No module named 'gym'`
    - `pip install gym==0.19.0`
    - however, LeggedRobot does not require gym, only for testing dreamerv3
3. `KeyError: 'anymal_c_flat'` (or any user defined task)
    - register a task in `legged_gym/envs/__init__.py` (see the file for example)
4. `[Error] [carb.windowing-glfw.plugin] GLFW initialization failed.
[Error] [carb.windowing-glfw.plugin] GLFW window creation failed!
[Error] [carb.gym.plugin] Failed to create Window in CreateGymViewerInternal`
    - `sudo apt-get install libglfw3-dev libglfw3`
    - Still unsolved


### I wonder whether it is OK to use PyTorch 2.0.0 with cuda-11.7.

1. install miniconda:
    - `bash Miniconda3-4.7.12-Linux-x86_64.sh`
2. Create a new python virtual env with python 3.8 (3.8 recommended). i.e. with conda:
    - `conda create -n isaacdreamer python==3.9`
    - `conda activate isaacdreamer`
3. Install pytorch 2.0.0 with cuda-11.7: (the new implement of dreamer depends on `torch.compile` and I am fear of changing the version). Most of the apex-gpu-servers has cuda-11.7 installed.
    - `pip3 install pytorch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0+cu117 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
4. Install IsaacGym, copy the isaacgym folder to your own path, i.e., ~/, and:
   - `cd isaacgym/python && pip install -e .`

### Train

I have no idea currently. In 2023.

### Test

I have no idea currently neither. In 2023.
