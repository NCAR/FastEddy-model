#!/bin/bash
#PBS -A <ProjectAccount>
#PBS -N FastEddy 
#PBS -l select=1:ncpus=4:mpiprocs=4:ngpus=4:mem=100GB
#PBS -l walltime=12:00:00
#PBS -q main 
#PBS -j oe
#PBS -l job_priority=economy

export BASEDIR=<path to code location>/FastEddy-model/
export SRCDIR=${BASEDIR}/SRC/FEMAIN
export TUTORIALDIR=${BASEDIR}/tutorials/
export EXAMPLE=<example .in filename>

hostname
module -t list
echo " "

mpiexec -n 4 --ppn 4 set_gpu_rank ${SRCDIR}/FastEddy ${TUTORIALDIR}/examples/${EXAMPLE}
