# DreamerV3-A1

### Install 

### Temporary I am using env isaacgym.

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


### Using PyTorch 2.0.0 with cuda-11.7, would it be OK?

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
