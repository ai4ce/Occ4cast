#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-849
#SBATCH --time=1:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=nusc
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=log/nusc_%A_%a.out
#SBATCH --error=log/nusc_%A_%a.err

sleep $(( (RANDOM%10) + 1 )) # to avoid issues when submitting large amounts of jobs

# Data path
DATA_PATH=/vast/xl3136/nuscenes/

# Output path
OUTPUT_PATH=/scratch/xl3136/dataset/OCFBench_nuScenes

# Start and end scene index
START=0

# Number of frames to aggregrate
A_PRE=70
A_POST=70

# Number of frames to predict
P_PRE=10
P_POST=10

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module purge
cd /scratch/$USER/Occ4cast/data_processing/nuscenes

singularity exec \
	    --overlay /scratch/$USER/environments/nusc.ext3:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; 
        python convert_nuscenes.py -d $DATA_PATH -o $OUTPUT_PATH -s $START -b ${SLURM_ARRAY_TASK_ID} --a_pre $A_PRE --a_post $A_POST --p_pre $P_PRE --p_post $P_POST"