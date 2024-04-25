# FastEddy® 
©2016 University Corporation for Atmospheric Research

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11042754.svg)](https://doi.org/10.5281/zenodo.11042754)

# Open-source License 
The FastEddy® model is licensed under the Apache License, Version 2.0 (the "License");
you may not use any source code in this repository except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# Description
FastEddy® (FE) is a large-eddy simulation (LES) model developed by the Research Applications Laboratory (RAL) at the U.S. National Science Foundation National Center for Atmospheric Research (NSF NCAR) in Boulder, Colorado, USA. The fundamental premise of FastEddy model development is to leverage the accelerated and more power efficient computing capacity of graphics processing units (GPU)s to enable not only more widespread use of LES in research activities but also to pursue the adoption of microscale and multiscale, turbulence-resolving, atmospheric boundary layer modeling into local scale weather prediction or actionable science and engineering applications.

## Contact
Please submit all comments, feedback, suggestions, or questions by email to the NSF NCAR FastEddy team at [fasteddy@ucar.edu](fasteddy@ucar.edu). Further information about FastEddy applications and research is available via the [RAL website](https://ral.ucar.edu/solutions/products/fasteddy). 

# Getting Started
To get started using FastEddy on NSF NCAR's Casper architecture simple instructions are provided below. These include a brief explanation of how to compile FastEddy, an example PBS job submission script, and a pointer to tutorial documentation for idealized test cases. Finally, reference publications for model formulation are provided.

## Beta-build
The Makefile-based build system included here assumes deployment on the NSF NCAR Casper system https://arc.ucar.edu/knowledge_base/70549550. FastEddy requires a C-compiler, MPI, and CUDA. On Casper ensure modules are loaded for openmpi, netcdf, and cuda with module -t list, and e.g. module load [intel or gnu/openmpi/cuda] as necessary. Currently, the default modules of intel, openMPI, and CUDA are loaded at login and suffice.

1. Navigate to SRC/FEMAIN
2. To build the FastEddy executable run make (optionally run make clean first if appropriate).

To build on other HPC systems with NVIDIA GPUs, check for availability of the aformentioned modules/dependencies. Successful compilation may require modifications to shell environment variable include or library paths, or alternatively minor adjustments to the include or library flags in SRC/FEMAIN/Makefile.    

## Example PBS job script
A bash-based PBS job submission script for running the model on NSF NCAR's Casper machine. This script assumes you have cloned this repository into a /glade/work/$USER/FastEddy directory you created.
```
#!/bin/bash
#
#PBS -N FastEddy 
#
# Replace "ProjectAccount" with your project account below 
#PBS -A ProjectAccount
#
#PBS -l select=1:ncpus=4:mpiprocs=4:ngpus=4:mem=100GB
#PBS -l gpu_type=v100
#PBS -l walltime=00:30:00
#PBS -q casper
#PBS -k oed
#
# Set environmental variables 
#
# Define the base and code directories in a non-purged filespace
export BASEDIR=/glade/work/$USER
export CODEDIR=$BASEDIR/FastEddy
#
# Define the source directory
export SRCDIR=$CODEDIR/SRC/FEMAIN
export EXAMPLEDIR=$CODEDIR/EXAMPLES
#
# Define and make the run directory in your scratch filespace (see Casper purge policy)
export CASEDIR=TEST/CBL
export RUNDIR=/glade/scratch/$USER/FastEddy/$CASEDIR
mkdir -p $RUNDIR
mkdir -p $RUNDIR/output
#
# Change directory to the run directory and copy the executable and the input file into it
cd $RUNDIR
\cp -u -p $SRCDIR/FastEddy .
\cp -u -p $EXAMPLEDIR/Example02_CBL.in .
#
#unload/load modules here if a non-default configuration was used in compilation
#e.g. module load gnu
# Output basic, often useful information about the compute node and runtime loaded modules 
hostname
pwd
module -t list
#
# RUN FastEddy
mpirun -n 4 ./FastEddy Example02_CBL.in
```

## Tutorials 
FastEddy tutorials for idealized cases are available at https://fasteddy-model.readthedocs.io

## References
Model Publications: 
1. FastEddy dry dynamics formulation, idealized case validation and performance benchmarks: https://doi.org/10.1029/2020MS002100
2. FastEddy moist dynamics extension and validation: https://doi.org/10.1029/2021MS002904  
