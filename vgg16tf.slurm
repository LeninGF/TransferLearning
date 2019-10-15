#!/bin/bash

#SBATCH -J myjob
#SBATCH -N 3
#SBATCH --gres=gpu:1
#SBATCH -p quinde-G9-2680
#SBATCH --qos=extended

hostname
python vgg16_tf2.py

echo "Training finished"
