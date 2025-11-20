#!/bin/bash
#PBS -A <ProjectAccount>
#PBS -N FE_convert 
#PBS -l select=2:ncpus=2:mpiprocs=2:mem=200GB
#PBS -l walltime=24:00:00
#PBS -q casper
#PBS -j oe
#PBS -l job_priority=economy

export BASEDIR=<path to code location>/FastEddy-model/
export SRCDIR=${BASEDIR}/scripts/python_utilities/post-processing/

hostname
module load conda
# The following conda environment (on Casper) can be created (for other platforms) from 
# the environment.yml file in this repository */scripts/batch_jobs/ directory
# with---> conda env create -f environment.yml
conda activate /glade/u/fehelp/casper/conda-envs/mpi4py-casper-oneapi-2024.2.1-openmpi-5.0.6 
which python

mpiexec python -u ${SRCDIR}/FEbinaryToNetCDF.py -f ${SRCDIR}/convert.json -a ${SRCDIR}/field_attributes.json
