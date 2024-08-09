#!/bin/bash
#PBS -A <ProjectAccount>
#PBS -N FastEddy 
#PBS -l select=1:ncpus=4:mpiprocs=4:ngpus=4:mem=100GB
#PBS -l walltime=12:00:00
#PBS -q casper
#PBS -j oe
#PBS -l job_priority=economy

export BASEDIR=<path to code location>/FastEddy-model/
export SRCDIR=${BASEDIR}/SRC/FEMAIN
export TUTORIALDIR=${BASEDIR}/tutorials/
export EXAMPLE=<example .in filename>

hostname
module -t list
echo " "

mpirun -np 4 ${SRCDIR}/FastEddy ${TUTORIALDIR}/examples/${EXAMPLE}
