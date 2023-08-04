#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=96GB
#SBATCH --job-name=4docc
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xl3136@nyu.edu
#SBATCH --output=log/4docc_%j.out
#SBATCH --error=log/4docc_%j.err
#SBATCH --gres=gpu:1

cd /scratch/$USER/Occ4D/baselines/4docc

singularity exec --nv \
	    --overlay /scratch/$USER/environments/occ4d.ext3:ro \
	    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	    /bin/bash -c "source /ext3/env.sh; 
        python train.py -d lyft \
            -r /vast/xl3136/lyft_kitti \
            --p_pre 5 \
            --p_post 5 \
            --batch-size 6 \
            --num-workers 16 \
            --num-epoch 15"