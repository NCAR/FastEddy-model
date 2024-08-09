#!/bin/bash
#PBS -A <ProjectAccount>
#PBS -N FE_convert 
#PBS -l select=2:ncpus=2:mpiprocs=2:mem=40GB
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -j oe
#PBS -l job_priority=economy

export BASEDIR=<path to code location>/FastEddy-model/
export SRCDIR=${BASEDIR}/scripts/python_utilities/post-processing/

hostname
module load conda
conda activate npl-2023b

mpirun -np 4 python -u ${SRCDIR}/FEbinaryToNetCDF.py ${SRCDIR}/convert.json
