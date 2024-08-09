*************
Release Notes
*************


FastEddy-model Version 2.0 Release Notes (20240809)
===================================================

.. dropdown:: Repository, build, and test

   * Three new tutorials (canopy, offshore, passive scalar transport and dispersion)
   * Demonstration of including terrain and N-ranks to N-files raw binary output and post-processing to single NetCDF via Python
   * Runtime parameter checks provide guidance for inter/intra-device parallelization settings  

.. dropdown:: Bugfixes

   * Linear interpolation of pressure bottom/top boundary conditions instead of constant
   * Small fix to the forcing term for condensation   
	
.. dropdown:: Enhancements

   * Additional explicit filters (divergence damping and horizontal 6th-order diffusion) [Contributed by Prof. Bowen Zhou from Nanjing University, China]
   * Two-equation canopy model
   * A suite of offshore roughness parameterizations
   * A dynamic formulation for thermal roughness length over land
   * Auxiliary passive scalar transport and dispersion
   * Additional documentation content (building, running on NSF NCAR HPC, publications, downloads, and more)  
  
	      
FastEddy-model Version 1.1 Release Notes (20240422)
===================================================

This is the initial release of the FastEddy, an NSF NCAR developed parallelized
and GPU-resident, large-eddy simulation code for accelerated modeling of the
atmospheric boundary layer.

In addition to the initial code, this release includes a patch for building
the system in the current NSF NCAR high performance computing environment on the
Casper and Derecho platforms, along with other changes as detailed below:

  .. dropdown:: Repository, build, and test

     * Add templates for Issues and Pull Requests
     * Set up the FastEddy Tutorial documentation
     * Consolidate FastEddy-tutorials content into FastEddy-model
     * Adjust FastEddy-tutorials BOMEX notebook & RTD Moist dynamics instructions for hosting datasets under new repo

  .. dropdown:: Bugfixes

     * Fix to the restart model capability
     * Clean compile with warnings addressed

  .. dropdown:: Enhancements

     * Accommodate building on Derecho


