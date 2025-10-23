**********
Real Cases
**********

Real cases are performed by dynamically downscaling from a mesoscale model such as WRF that can provide initial and boundary conditions for a one-way nested FastEddy simulation. These mesoscale-LES coupled simulations require the following preprocessing steps:

* Step 1: **GeoSpec**. Georeference specification step. Expects a NetCDF-formatted file of location-specific, georeferenced coordinate frame (lat/lon), projected Cartesian coordinate frame (x,y), elevation and land cover to establish a new NetCDF file of reference geolocated domain static characteristics specification including mapping of land cover category to roughness length.
* Step 2: **SimGrid**. Simulation grid definition step. Defines a FastEddy gridded domain at a specificed grid spacing, location and extent using the file resulting from ther previous GeoSpec step and a FastEddy input parameters file (with targeted domain configuration parameters) as inputs.
* Step 3: **GenICBCs**. Generate initial conditions/boundary (ICBCs) conditions step. Creates ICBCs for a targeted FastEddy domain (defined in the SimGrid step) from a set of mesoscale model results. 

The following tutorial provides a practical example of performing these preprocessing steps followed by a corresponding weather-driven FastEddy simulation for a real-world downscaled scenario. 

.. toctree::

   cases_real/WRF_coupling_case0.rst
   cases_real/WRF_coupling_case0_FE.rst
