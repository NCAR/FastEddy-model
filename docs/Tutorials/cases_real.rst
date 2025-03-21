***************
Real Test Cases
***************

Real cases are dynamically downscaled from a mesoscale model such as WRF that provides initial and boundary conditions for the one-way nested FastEddy simulation. These mesoscale-LES coupled simulations require the following preprocessing steps:

* Step 1: **GeoSpec**. Reads in GIS information about terrain elevation and land cover and creates a reference netCDF file.
* Step 2: **SimGrid**. Defines a FastEddy domain of a specificed grid spacing, location and extentent using GeoSpec file and a FastEddy parameters file as inputs.
* Step 3: **GenICBCs**. Creates initial and boundary conditions (ICBCs) for FastEddy from a mesoscale run over the SimGrid generated domain. 

The following tutorial provides a practical example of how to run the 3 preprocessing steps and the corresponding weather-driven FastEddy simulation for a real case.

.. toctree::

   cases_real/WRF_coupling_case0.rst
