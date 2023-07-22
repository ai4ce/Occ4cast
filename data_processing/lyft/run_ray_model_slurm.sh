#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=160GB
#SBATCH --job-name=lyft
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=log/lyft_%j.out
#SBATCH --error=log/lyft_%j.err

START=0
END=1
A_PRE=70
A_POST=70
P_PRE=10
P_POST=10
INPUT=/vast/xl3136/lyft_kitti/train

module purge
cd /scratch/$USER/Occ4D/data_processing/lyft

singularity exec --nv \
	    --overlay /scratch/$USER/environments/lyft.ext3:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh;
        python ray_model_vis.py -v -i $INPUT -s $START -e $END --a_pre $A_PRE --a_post $A_POST --p_pre $P_PRE --p_post $P_POST"