.. _run_fasteddy_amdgpu:

**************************************************
Building FastEddy on AMD GPU Accelerated platforms
**************************************************

These instructions will help users get started with building FastEddy for systems with AMD GPU accelerators. This can be beneficial for users who have allocations on 

   * `Oak Ridge National Laboratory's Frontier <https://www.olcf.ornl.gov/frontier/>`_
   * `Pawsey Supercomputing Centre's Setonix <https://pawsey.org.au/systems/setonix/>`_
   * `CSCS's Lumi <https://www.lumi-supercomputer.eu/may-we-introduce-lumi/>`_
   * `Fluid Numerics' Galapagos <https://galapagos.fluidnumerics.com>`_


Compilation
===========

FastEddy requires a C-compiler, MPI, and CUDA. 

   1. Download the source code from the `Releases <https://github.com/NCAR/FastEddy-model/releases>`_ page and unpack the release in the desired location or clone the `repository <https://github.com/NCAR/FastEddy-model>`_ in the desired location.

   2. Navigate to the **SRC/FEMAIN** directory.

   3. To build the FastEddy executable run :code:`make -f Makfile.hip` (optionally run :code:`make clean` first if appropriate).

You may need to define a few environment variables that influence the build process to properly set the paths to various dependencies and to select the target GPU.

   * :code:`ROCM_PATH` : This is the path to your ROCm installation. This variable defaults to :code:`/opt/rocm`. However, on some systems, multiple versions of ROCm may be available via environment modules and this variable may need to be adjusted accordingly.
   * :code:`MPI_ROOT` : This is the path to your MPI installation. Since this environment variable is not necessarily defined through an HPC center's environment modules, it is recommended that you set this variable appropriately.
   * :code:`NETCDF_C_ROOT`: This is the path to your NetCDF-C installation. Since this environment variable is not necessarily defined through an HPC center's environment modules, it is recommended that you set this variable appropriately.
   * :code:`GPU_ARCH` : This is the AMD GPU architecture code for the target GPU you want to build for. This variable defaults to :code:`gfx90a`, which corresponds to the MI210, MI250, and MI250X GPUs.

The :code:`FastEddy` executable will be located in the **SRC/FEMAIN** directory. To
build on other HPC systems with NVIDIA GPUs, check for availability of the aformentioned
modules/dependencies. Successful compilation may require modifications to shell environment
variable include or library paths, or alternatively minor adjustments to the include or library
flags in **SRC/FEMAIN/Makefile.hip**.
