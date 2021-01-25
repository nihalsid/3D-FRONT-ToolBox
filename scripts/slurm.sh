#!/bin/bash

#SBATCH --job-name export-meshes
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --exclude=moria

cd /rhome/ysiddiqui/3D-FRONT-ToolBox/scripts
python json2obj.py --future_path /cluster/gondor/mdahnert/datasets/future3d/3D-FUTURE-model/ --json_path /cluster/gondor/mdahnert/datasets/front3d/3D-FRONT/ --save_path /cluster/gondor/ysiddiqui/3DFrontMeshes --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID
