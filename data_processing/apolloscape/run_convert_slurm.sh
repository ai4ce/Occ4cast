#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-89
#SBATCH --time=1:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=apollo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=log/apollo_%A_%a.out
#SBATCH --error=log/apollo_%A_%a.err

# Data path
DATA_PATH=/media/xinhao/DATA/apolloscape

# Output path
OUTPUT_PATH=/home/xinhao/dataset/apolloscape

# Start and end scene index
START=0
END=1

# Number of frames to aggregrate
A_PRE=70
A_POST=70

# Number of frames to predict
P_PRE=10
P_POST=10

python Occ4D_apollo.py -d $DATA_PATH -o $OUTPUT_PATH -s $START -e $END --a_pre $A_PRE --a_post $A_POST --p_pre $P_PRE --p_post $P_POST