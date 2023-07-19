#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=lyft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=log/lyft_%j.out
#SBATCH --error=log/lyft_%j.err

# Data path
DATA_PATH=/lyft/train

# Output path
OUTPUT_PATH=/vast/xl3136/lyft_kitti/train

# Start and end scene index
START=0
END=180

# Number of frames to aggregrate
A_PRE=70
A_POST=70

# Number of frames to predict
P_PRE=10
P_POST=10

module purge
cd /scratch/$USER/Occ4D/data_processing/lyft

singularity exec \
	    --overlay /scratch/$USER/environments/lyft.ext3:ro \
        --overlay /scratch/$USER/dataset/lyft/lyft.sqf:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; 
        python convert_lyft.py -m -d $DATA_PATH -o $OUTPUT_PATH -s $START -e $END --a_pre $A_PRE --a_post $A_POST --p_pre $P_PRE --p_post $P_POST"