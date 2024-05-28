#!/bin/bash -l



module load cudnn/8.1.0.77/cuda-11.2
module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python/3.9.6-gnu-10.2.0
module load cuda/11.2.0/gnu-10.2.0
module load cudnn/8.1.0.77/cuda-11.2
module load tensorflow/2.11.0/gpu

module list

#$ -N MolGans

#$ -l gpu=2

#$ -l h_rt=0:50:0

#$ -l mem=1G

#$ -l tmpfs=15G

#$ -wd /home/ucaqaad/Scratch/Projector

cd $TMPDIR

cd /home/ucaqaad/Scratch/Projector

python MolGans.py

tar zcvf $HOME/Scratch/files_from_job_$JOB_ID.tar.gz $TMPDIR
