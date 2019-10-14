#!/bin/bash

#SBATCH -J myjob
#SBATCH -N 3
#SBATCH --gres=gpu:1
#SBATCH -p quinde-G9-2680

hostname
python vgg16_tf.py

echo "Training finished"
