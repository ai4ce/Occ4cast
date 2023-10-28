#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-849
#SBATCH --time=1:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=raymodel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=log/raymodel_%A_%a.out
#SBATCH --error=log/raymodel_%A_%a.err
#SBATCH --gres=gpu:1

START=0
A_PRE=70
A_POST=70
P_PRE=10
P_POST=10
INPUT=/scratch/xl3136/dataset/OCFBench_nuScenes

module purge
cd /scratch/$USER/Occ4cast/data_processing/raymodel

singularity exec --nv \
	    --overlay /scratch/$USER/environments/nusc.ext3:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh;
        python ray_model.py -i $INPUT -s $START -b ${SLURM_ARRAY_TASK_ID} --a_pre $A_PRE --a_post $A_POST --p_pre $P_PRE --p_post $P_POST"