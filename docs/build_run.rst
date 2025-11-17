.. _run_fasteddy:

*************
Build and Run
*************

These instructions will help users get started running FastEddy in the
`NSF NCAR High Performance Computing (HPC) environment <https://ncar-hpc-docs.readthedocs.io/en/latest/>`_,
using `Derecho <https://arc.ucar.edu/knowledge_base/74317833>`_ and
`Casper <https://arc.ucar.edu/knowledge_base/70549550>`_.

Existing Builds
===============

The FastEddy team has installed the FastEddy software on both Casper and
Derecho.

If desired, users can skip the compilation step and run the following
commands to load FastEddy. 

.. note::

   Users need to replace *<version>* below with the desired X.Y.Z
   version to load (e.g. 4.0.0, 3.0.0, etc.).


Casper
------

.. code-block:: shell

  module use /glade/u/fehelp/casper/installations/modulefiles
  module load fasteddy/<version>

Derecho
-------

.. code-block:: shell

  module use /glade/u/fehelp/derecho/installations/modulefiles
  module load fasteddy/<version>

Compilation
===========

FastEddy requires a C-compiler, MPI, and CUDA.

Whether compiling on NSF NCAR HPC Environments or other platforms, users
should do the following:

   1. Download the source code from the
      `Releases <https://github.com/NCAR/FastEddy-model/releases>`_ page
      and unpack the release in the desired location or clone the
      `repository <https://github.com/NCAR/FastEddy-model>`_ in the desired location.

   2. Navigate to the **SRC/FEMAIN** directory.


Compilation on NSF NCAR HPC Environments
----------------------------------------

These instructions will help users get started with building FastEddy in the 
`NSF NCAR High Performance Computing (HPC) environment <https://ncar-hpc-docs.readthedocs.io/en/latest/>`_,
using `Derecho <https://arc.ucar.edu/knowledge_base/74317833>`_ and
`Casper <https://arc.ucar.edu/knowledge_base/70549550>`_.

The Makefile-based build system included here assumes deployment on the NSF
NCAR HPCs. 

Currently, the default modules loaded at login suffice on Casper, however the
:code:`cuda` module will need to be loaded on Derecho by running
:code:`module load cuda`.

After following steps 1 and 2 above, build the FastEddy executable by running:

.. code::

   make

(Optionally, run :code:`make clean` first if appropriate.)

To enable model extensions, additional compilation flags are required:

* To include the **Generalized Actuator Disk (GAD)** parameterized model, run:

  .. code::

     make WITH_GAD=1

* To include the **building-resolving URBAN** model extension, run:

  .. code::

     make WITH_URBAN=1

* To enable **both** extensions, run:

  .. code::

     make WITH_GAD=1 WITH_URBAN=1

The :code:`FastEddy` executable will be located in the **SRC/FEMAIN** directory.

Compilation on AMD GPU Accelerated Platforms
--------------------------------------------

These instructions will help users get started with building FastEddy for
systems with AMD GPU accelerators. This can be beneficial for users who
have allocations on: 

   * `Oak Ridge National Laboratory's Frontier <https://www.olcf.ornl.gov/frontier/>`_
   * `Pawsey Supercomputing Centre's Setonix <https://pawsey.org.au/systems/setonix/>`_
   * `CSCS's Lumi <https://www.lumi-supercomputer.eu/may-we-introduce-lumi/>`_
   * `Fluid Numerics' Galapagos <https://galapagos.fluidnumerics.com>`_

After following steps 1 and 2 above, build the FastEddy executable by running:

.. code::

   make -f Makefile.hip

(Optionally, run :code:`make clean` first if appropriate.)

To enable model extensions, additional compilation flags are required:

* To include the **Generalized Actuator Disk (GAD)** parameterized model, run:

  .. code::

     make -f Makefile.hip WITH_GAD=1                                                                                                                                                                                        
* To include the **building-resolving URBAN** model extension, run:

  .. code::

     make -f Makefile.hip WITH_URBAN=1

* To enable **both** extensions, run:

  .. code::

     make -f Makefile.hip WITH_GAD=1 WITH_URBAN=1 

Users may need to define a few environment variables that influence the build
process to properly set the paths to various dependencies and to select the
target GPU.

   * :code:`ROCM_PATH` : This is the path to your ROCm installation. This variable
     defaults to :code:`/opt/rocm`. However, on some systems, multiple versions of
     ROCm may be available via environment modules and this variable may need to be
     adjusted accordingly.
   * :code:`MPI_ROOT` : This is the path to your MPI installation. Since this
     environment variable is not necessarily defined through an HPC center's
     environment modules, it is recommended that you set this variable appropriately.
   * :code:`NETCDF_C_ROOT`: This is the path to your NetCDF-C installation. Since
     this environment variable is not necessarily defined through an HPC center's
     environment modules, it is recommended that you set this variable appropriately.
   * :code:`GPU_ARCH` : This is the AMD GPU architecture code for the target GPU you
     want to build for. This variable defaults to :code:`gfx90a`, which corresponds
     to the MI210, MI250, and MI250X GPUs.

The :code:`FastEddy` executable will be located in the **SRC/FEMAIN** directory. To
build on other HPC systems with NVIDIA GPUs, check for availability of the aformentioned
modules/dependencies. Successful compilation may require modifications to shell environment
variable include or library paths, or alternatively minor adjustments to the include or
library flags in **SRC/FEMAIN/Makefile.hip**.

Example PBS Run Scripts
=======================

Below is bash-based PBS job submission script for running the model on NSF NCAR's Casper
and Derecho. These example scripts:

* fasteddy_pbs_script_casper.sh
* fasteddy_pbs_script_derecho.sh

can be found in the **scripts/batch_jobs** subdirectory of the
`GitHub FastEddy-model repository <https://github.com/NCAR/FastEddy-model>`_.

The FastEddy code will write its output to an **output** directory. Please create an output
directory, if one does not already exist, in the same location as this script.

The code will produce an **log file** with the name *FastEddy.o<job_id>*
(for example, *FastEddy.o4960197*) in the current working directory.

To submit the script for batch processing, run :code:`qsub <name of script>`, replacing
*<name of script>* with the name of the script.

.. note::

   In the scripts, users will need to:
     * Replace "<ProjectAccount>" below with a valid Project Account.
     * Replace "<path to code location>" below with the location that contains the **FastEddy-model** directory from unpacking the release.
     * Replace "<example .in filename>" below with the name of the example .in filename.  For example, *Example01_NBL.in*, *Example02_CBL.in*, etc.

Casper
------

.. literalinclude:: ../scripts/batch_jobs/fasteddy_pbs_script_casper.sh

Derecho
-------

.. literalinclude:: ../scripts/batch_jobs/fasteddy_pbs_script_derecho.sh
