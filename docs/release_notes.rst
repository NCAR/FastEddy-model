*************
Release Notes
*************

FastEddy-model Version 4.0 Release Notes (20251023)
===================================================

.. dropdown:: Repository, build, and test

   * Two new tutorials for coupled mesoscale-LES real-world case including urban structure impacts and an idealized demonstration of wind farm flow under a wind shift
   * New comprehensive model parameter reference guide
   * Documentation updates to organization and content clarity
   * IO module feature extensions providing output field attributes (units and description) in both netCDF and raw binary forms
   * Modular framework and Makefile-based build process for model extensions (GAD and URBAN)  

.. dropdown:: Bugfixes

   * Fix to the Coriolis force term to properly pass down parameters to the GPU
   * Introduction of limiters for instantaneous heat and moisture exchange coefficients to avoid unrealistically large values in surface-layer fluxes

.. dropdown:: Enhancements

   * Urban model using an immersed body force method (IBFM) (urbanSelector=1)
   * Wind turbine model using a generalized actuator disk (GAD) method (GADSelector=1)

FastEddy-model Version 3.0 Release Notes (20250415)
===================================================

.. dropdown:: Repository, build, and test

   * One new tutorial for coupled mesoscale-LES real-world case
   * Additional makefile for building for AMD GPUs with hip and corresponding documentation [Contributed by Dr. Joe Schoonover from Fluid Numerics]


.. dropdown:: Bugfixes

   * Inclusion of omitted J13 and J23 metric tensor terms in the coordinate transformation
   * Fix to the restart capability including T&D of auxiliary scalars
   * Some fixes to the surface layer model. Definition of the heat and moisture exchange coefficients, range of stability corrections, and formulation of integrated stability functions for stable conditions  

.. dropdown:: Enhancements

   * Offline one-way coupling of FastEddy to a mesoscale NWP model (hydroBCs=1)
   * Pre-processing python utilities (GeoSpec, SimGrid, and GenICBCs) for real-world mesoscale-LES simulations
   * Local filter for terrain in pre-processing SimGrid step
     [Contributed by Eloisa Raluy-López from University of Murcia, Spain]
   * Cell perturbation method to instigate formation of fully developed turbulence in coupled mesoscale-LES simulations (cellpertSelector = 1)
   * Added a scalar float variable “time” to output files to indicate simulated time since beginning of simulation in seconds  

     
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


