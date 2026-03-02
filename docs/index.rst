==================
FastEddy\ :sup:`®`
==================

.. image:: _static/CoastalCase_u_y-32km_0440.png

FastEddy\ :sup:`®` (FE) is a large-eddy simulation (LES) model developed by the
Research Applications Laboratory (RAL) at the U.S. National Science
Foundation National Center for Atmospheric Research (NSF NCAR) in Boulder,
Colorado, USA. The fundamental premise of FastEddy model development is to
leverage the accelerated and more power efficient computing capacity of
graphics processing units (GPU)s to enable not only more widespread use of
LES in research activities but also to pursue the adoption of microscale
and multiscale, turbulence-resolving, atmospheric boundary layer
modeling into local scale weather prediction or actionable science and
engineering applications.

Citations
---------

The FastEddy code is located in an open, public
`GitHub FastEddy-model repository <https://github.com/NCAR/FastEddy-model>`_.
Please cite FastEddy as follows:

  | Sauer, J., and D. Muñoz-Esparza. "The FastEddy resident-GPU accelerated large-eddy
  |   simulation framework: model formulation, dynamical-core validation and performance
  |   benchmarks". *Journal of Advances in Modeling Earth Systems*, vol. 12 (2020)
  |   https://doi.org/10.1029/2020MS002100


Contributing Authors
--------------------

The following authors have contributed to this documentation:

FastEddy: NSF-NCAR Research Applications Laboratory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
web: https://ral.ucar.edu/, email: fasteddy@ucar.edu

  * Jeremy Sauer 
  * Domingo Muñoz-Esparza
  * Julie Prestopnik
  * Eric Hendricks

Building FastEddy on AMD GPUs: Fluid Numerics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
web: https://fluidnumerics.com, email: support@fluidnumerics.com

  * Joe Schoonover 

  
.. toctree::
   :hidden:

   release_notes.rst
   downloads.rst
   build_run.rst
   Tutorials/index
   parameters.rst
   publications.rst
