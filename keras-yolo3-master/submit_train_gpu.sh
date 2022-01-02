#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J Train_GPU
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
#BSUB -gpu "num=2:mode=exclusive_process" 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=16GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u nnho@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o OutputGPU_Tirs_%J.out 
#BSUB -e ErrorGPU_Tirs_%J.err

module load python3/3.7.11
module load cuda/7.0
cd Desktop/DeepLearning/keras-yolo3-master
echo "running GPU training"
python3 train.py -c config_gpu.json

