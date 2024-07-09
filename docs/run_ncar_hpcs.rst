.. _run_fasteddy:

**************************
Running under NSF NCAR HPC
**************************

These instructions will help users get started running FastEddy in the 
`NSF NCAR High Performance Computing (HPC) environment <https://ncar-hpc-docs.readthedocs.io/en/latest/>`_,
using `Derecho <https://arc.ucar.edu/knowledge_base/74317833>`_ and
`Casper <https://arc.ucar.edu/knowledge_base/70549550>`_.

Compilation
===========

The Makefile-based build system included here assumes deployment on the NSF
NCAR HPCs. FastEddy requires a C-compiler, MPI, and CUDA. Currently, the
default modules loaded at login suffice on Casper, however the :code:`cuda` module
will need to be loaded on Derecho by running :code:`module load cuda`.

   1. Download the source code from the `Releases <https://github.com/NCAR/FastEddy-model/releases>`_ page and unpack the release in the desired location or clone the `repository <https://github.com/NCAR/FastEddy-model>`_ in the desired location.

   2. Navigate to the **SRC/FEMAIN** directory.

   3. To build the FastEddy executable run :code:`make` (optionally run :code:`make clean` first if appropriate).

The :code:`FastEddy` executable will be located in the **SRC/FEMAIN** directory. To
build on other HPC systems with NVIDIA GPUs, check for availability of the aformentioned
modules/dependencies. Successful compilation may require modifications to shell environment
variable include or library paths, or alternatively minor adjustments to the include or library
flags in **SRC/FEMAIN/Makefile**.

Example PBS job script for Casper
=================================

Below is bash-based PBS job submission script for running the model on NSF NCAR's Casper machine.
The FastEddy code will write its output to an **output** directory. Please create an output
directory, if one does not already exist, in the same location as this script.

Sample Script
-------------

.. note::

   * Replace "<ProjectAccount>" below with a valid Project Account.
   * Replace "<path to code location>" below with the location that contains the **FastEddy-model** directory from unpacking the release.
   * Replace "<example .in filename>" below with the name of the example .in filename.  For example, *Example01_NBL.in*, *Example02_CBL.in*, etc.
   
.. code-block:: bash

  #!/bin/bash
  #PBS -A <ProjectAccount>
  #PBS -N FastEddy 
  #PBS -l select=1:ncpus=4:mpiprocs=4:ngpus=4:mem=100GB
  #PBS -l walltime=12:00:00
  #PBS -q casper
  #PBS -r n 
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

The code will produce an log file with the name *FastEddy.o<job_id>*
(for example, *FastEddy.o4960197*) in the current working directory.

To submit the script for batch processing, run `qsub <name of script>`, replacing
*<name of script>* with the name of the script.

Example PBS job script for Derecho
==================================

Below is bash-based PBS job submission script for running the model on NSF NCAR's Derecho machine.
The FastEddy code will write its output to an `output` directory. Please create an output
directory, if one does not already exist, in the same location as this script.

Sample Script
-------------

.. note::

   * Replace "<ProjectAccount>" below with a valid Project Account.
   * Replace "<path to code location>" below with the location that contains the `FastEddy-model` directory from unpacking the release.
   * Replace "<example .in filename>" below with the name of the example .in filename.  For example, Example01_NBL.in, Example02_CBL.in, etc.

.. code-block:: bash

  #!/bin/bash
  #PBS -A <ProjectAccount>
  #PBS -N FastEddy 
  #PBS -l select=1:ncpus=4:mpiprocs=4:ngpus=4:mem=100GB
  #PBS -l walltime=12:00:00
  #PBS -q main 
  #PBS -r n 
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

The code will produce an log file with the name *FastEddy.o<job_id>*
(for example, *FastEddy.o4960197*) in the current working directory.

To submit the script for batch processing, run `qsub <name of script>`, replacing
*<name of script>* with the name of the script.

   
