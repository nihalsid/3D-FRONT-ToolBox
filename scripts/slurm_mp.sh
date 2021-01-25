#!/bin/bash

#SBATCH --job-name export-meshes
##SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --exclude=moria

cd /rhome/ysiddiqui/3D-FRONT-ToolBox/scripts
python mp_sdf_to_mesh.py --sdf_dir /cluster_HDD/sorona/adai/data/matterport/mp_sdf_vox_1cm_color-complete --output_dir /cluster/gondor/ysiddiqui/Matterport3DMeshes --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID