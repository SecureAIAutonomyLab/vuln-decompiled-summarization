# ARC Video Guide
https://youtu.be/PXJh-3_tDX0

# Disclaimer
The information provided in the video is outdated, and wasn't aimed at just teaching ARC (it was more centered around a development environment for ARC). If you understand conda, and have a working development  environment already, then you don't need to watch the video. Just follow this guide

Moreover, the video demonstrates only one method for executing code on the ARC system. Personally, I prefer to develop locally on my system and maintain an ARC session within a terminal in Visual Studio Code. I use Git to synchronize code changes from my local environment to ARC. While it's possible to access ARC in a more direct fashion via remote VS Code, I choose not to do so to avoid over-allocating resources in ARC, as this is typically what you find in the Industry (AWS has multi-gpu instances that are 40 bucks an hour, you can find resources cheaper than that, but all of them are pretty pricey). By interacting with ARC solely through a terminal, I can efficiently manage resource usage and continue debugging and testing without unnecessarily occupying ARC resources. However, if the issues you face are solely related to loading in a bigger model, then some debugging/testing on ARC is unavoidable. But when you understand the ARC environment and get used to it, you'll just be submitting Slurm/batch jobs in anyway, and there will be no interactive sessions in ARC that are needed.


# Logging into arc and starting an interactive session with a single v100 GPU node
```bash
ssh abc123@arc.utsa.edu
srun -p gpu1v100 -n 1 -t 01:30:00 -c 40 --pty bash

# make sure your code goes to /work/abc123 dir (it can also goto /home/abc123 if you want)
# but your data MUST go to /work/abc123
cd /work/abc123

git clone https://github.com/SecureAIAutonomyLab/python-package.git
cd python-package # you will now be ready to set up a conda environment
```

# A common connection issue with ARC
```bash
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!

Host key verification failed.
```
If you encounter a "Host key verification failed" error when connecting via SSH, indicating the remote host identification has changed, you need to update your known_hosts file. First, open this file located at C:\Users\your_username\.ssh\known_hosts with a text editor like Notepad. Locate and delete the line that mentions the server you're trying to connect to, often specified by line number in the error message. This step removes the outdated server key. Upon your next SSH connection attempt, you'll be prompted to accept the new host key, thereby updating your known_hosts file and resolving the issue. This error typically occurs due to legitimate server changes or reconfigurations.


# Creating new conda environment, and then installing what you need manually
```bash
module load anaconda3

# You may need to do exec bash after loading the module to see the base conda environment prefix
# I'm not sure because I use a different approach, and install miniconda more directly, which involes more steps

conda create -n my_env python=3.11
conda activate my_env

# install pytorch, go here for other versions (https://pytorch.org/get-started/locally/)
# ARC GPU Nodes have Nvidia/CUDA Toolkit Version = 12.0, which means we need to compile pytorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c huggingface transformers
conda install -c conda-forge datasets

pip install -r requirements.txt # If you have a requirements.txt going
```
```
# Need to run this everytime you log in, unelss you add the export to your ~/.bashrc
# Important to avoid exceeding disk quota
export CONDA_PKGS_DIRS=/work/zwe996/.conda/pkgs && \
mkdir -p /work/zwe996/.conda/pkgs
```
# In general, if you run into a `disk quota exceed` error it just means you need to change an environment variable.
For example, anytime you download very large models or datasets, you will likely have to make a change to avoid too much data being temporarily cached in your home directory.
You need to delete the old HF cache in your home directly, and update the environment variable that points to that directory.
```bash
# Need to run this everytime you log in, unelss you add the export to your ~/.bashrc
export HF_HOME=/work/zwe996/.cache/huggingface && \
mkdir -p /work/zwe996/.cache/huggingface
```

# Creating existing conda environment from `environment.yml` file
Use this section if you already have a conda `environment.yml` file that specifys all your dependencies

```bash
conda env create --prefix ./env --file environment.yml
conda activate ./env
```

Now your environment is set up, and you are ready to run your code.

# Guide for Slurm Scripting
This portion of the guide is only for those wanting to go above and beyond on ARC. It shows the basics of submitting batch jobs on Slurm. All HPC environments that manage many distributed resources will have something like this in place. The basic idea is that users submit jobs, and those jobs are queued, and when resources are available and they are next in line, the job runs, and the results are saved.

For long-running jobs in particular, you may want to consider using a slurm script. However, before using a slurm script, you should make sure to set up your computing environment in an interactive session (as shown above with `srun`) to make sure that your code will run (set up your conda environment, move your data to `/work/abc`, etc).

Below is a sample script which covers the following steps:

1. Requests resources from Slurm
2. Activates a conda environment
3. Executes Python code

Example `job_script.slurm`  
Note: this may not work for you as your setup is likely different. For example, you'll likely have to do a `module load anaconda3` and then activate your conda environment. You'll need to check the documentation found at the bottom of this guide.
```bash
#!/bin/bash
#SBATCH --job-name=python_job       # Create a short name for your job, required
#SBATCH --nodes=1                   # Number of nodes, required
#SBATCH --ntasks=40                 # Number of CPU cores to use, required
#SBATCH --time=01:00:00             # Time limit hrs:min:sec, required
#SBATCH --output myjob.o%j          # Name of the stdout output file, required
#SBATCH --error myjob.e%j           # Name of stderr error file 
#SBATCH --partition=gpu1v100        # Specify the name of the GPU partition, required

eval "$(conda shell.bash hook)"
conda activate /work/abc123/my_env/  # Activate your conda environment from dir
export HF_HOME=/work/abc123/.cache/huggingface # You may not need to do these exports
export CONDA_PKGS_DIRS=/work/abc123/.conda/pkgs # Only if you're having disk quota exceeded errors

python /work/abc123/python-package/script.py      # Execute your Python script
```
run the script by doing:

`[abc123@login003 ~]$ sbatch job_script.slurm`

Once you've started the job, see more documentation here to view its progress.

https://hpcsupport.utsa.edu/foswiki/pub/ARC/WebHome/Running_Jobs_On_Arc.pdf

# Arc Documentation Link

https://hpcsupport.utsa.edu/foswiki/bin/view/ARC/WebHome
