**************************
Parameters Reference Guide
**************************

.. |br| raw:: html

   <br/>

This page defines the configurable parameters available in FastEddy\ :sup:`®`.
Parameters are organized into logical sets by model feature or model configuration component.

Each table provides the:

  * **Parameter name:** Description
  * Minimum value (if applicable)
  * Maximum value (if applicable)
  * Default value (if applicable)
  * Req. (Requirement) 
    
    * M (Mandatory)
    * O (Optional)
    * C-M (Conditionally Mandatory, with *the condition listed in the description*)
    * C-O (Conditionally Optional, with *the condition listed in the description*)

These definitions serve as a reference to ensure correct configuration and valid inputs for FastEddy simulations.

FEMPI
=====

.. csv-table::
   :file: csv/fempi.csv
   :header-rows: 1
   :delim: ,
   :class: longtable

FECUDA
======

.. csv-table::
   :file: csv/fecuda.csv
   :delim: ,
   :class: longtable

IO 
===

.. csv-table::
   :file: csv/io.csv
   :delim: ,
   :class: longtable

	   
GRID
====
.. csv-table::
   :file: csv/grid.csv
   :delim: ,
   :class: longtable

TIME INTEGRATION
================

.. csv-table::
   :file: csv/time_integration.csv
   :delim: ,
   :class: longtable

	   
HYDRO CORE
==========

BOUNDARY CONDITIONS
-------------------

.. csv-table::
   :file: csv/hydro_core_boundary_cond.csv
   :delim: ,
   :class: longtable

HYDRO_IO/LOGGING
----------------

.. csv-table::
   :file: csv/hydro_core_io_logging.csv
   :delim: ,
   :class: longtable

ADVECTION
---------

.. csv-table::
   :file: csv/hydro_core_advection.csv
   :delim: ,
   :class: longtable

MOISTURE
--------

.. csv-table::
   :file: csv/hydro_core_moisture.csv
   :delim: ,
   :class: longtable

CORIOLIS
--------

.. csv-table::
   :file: csv/hydro_core_coriolis.csv
   :delim: ,
   :class: longtable

TURBULENCE
----------

.. csv-table::
   :file: csv/hydro_core_turbulence.csv
   :delim: ,
   :class: longtable

CANOPY
------

.. csv-table::
   :file: csv/hydro_core_canopy.csv
   :delim: ,
   :class: longtable

DIFFUSION
---------

.. csv-table::
   :file: csv/hydro_core_diffusion.csv
   :delim: ,
   :class: longtable

AUXILIARY SCALARS AND SOURCES
-----------------------------

.. csv-table::
   :file: csv/hydro_core_auxiliary_scalars.csv
   :delim: ,
   :class: longtable

EXPLICIT FILTERS
----------------

.. csv-table::
   :file: csv/hydro_core_explicit_filters.csv
   :delim: ,
   :class: longtable

RAYLEIGH DAMPING LAYER
----------------------

.. csv-table::
   :file: csv/hydro_core_rayleigh_damping_layer.csv
   :delim: ,
   :class: longtable

SURFACE LAYER
-------------

.. csv-table::
   :file: csv/hydro_core_surface_layer.csv
   :delim: ,
   :class: longtable

CELL PERTURBATION METHOD
------------------------

.. csv-table::
   :file: csv/hydro_core_cell_perturbation.csv
   :delim: ,
   :class: longtable

BASE-STATE
----------

.. csv-table::
   :file: csv/hydro_core_base_state.csv
   :delim: ,
   :class: longtable

LARGE SCALE FORCINGS
--------------------

.. csv-table::
   :file: csv/hydro_core_large_scale_forcings.csv
   :delim: ,
   :class: longtable

EXTENSIONS
==========

GAD
---

.. csv-table::
   :file: csv/extensions_gad.csv
   :delim: ,
   :class: longtable

URBAN
-----

.. csv-table::
   :file: csv/extensions_urban.csv
   :delim: ,
   :class: longtable
