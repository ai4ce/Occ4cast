#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-52
#SBATCH --time=1:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=apollo
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=log/apollo_%A_%a.out
#SBATCH --error=log/apollo_%A_%a.err

# Data path
DATA_PATH=/vast/xl3136/apolloscape

# Output path
OUTPUT_PATH=/vast/xl3136/apolloscape_kitti

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
cd /scratch/$USER/Occ4D/data_processing/apolloscape

singularity exec \
	    --overlay /scratch/$USER/environments/apollo.ext3:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; 
        python convert_apollo_batch.py -d $DATA_PATH -o $OUTPUT_PATH -b ${SLURM_ARRAY_TASK_ID} -s $START --a_pre $A_PRE --a_post $A_POST --p_pre $P_PRE --p_post $P_POST"