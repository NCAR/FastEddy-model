Introduction
============

`FastEddy`_ is a resident-GPU large eddy simulation (LES) model owned by the National Center for Atmospheric Research (`NCAR`_) Research Applications Laboratory (`RAL`_). It is designed for future turbulence-resolving numerical weather prediction. 

.. _FastEddy: https://ral.ucar.edu/solutions/products/fasteddy
.. _NCAR: https://ncar.ucar.edu
.. _RAL: https://ral.ucar.edu

This is a tutorial designed so that a user can learn how to execute FastEddy. Four test cases are described: 

* Dry neutral boundary layer
* Dry convective boundary layer
* Dry stable boundary layer
* Moist cloud-topped boundary layer

Required tutorial resources including input files, python utilities and Jupyter Notebooks are provided in https://github.com/NCAR/FastEddy-tutorials. All test cases are ideal setups over zero terrain. For each case, the user will set up the input parameter file, execute FastEddy, visualize the output using a Jupyter notebook, and perform some basic analysis of the output. After examining the test cases, the user will carry out some sensitivity tests by changing various input parameters. The purpose of these tests are for the user to become more familiar with the input parameters, and how changes to those parameters affect the output. After the tutorial, the user is expected to have basic knowledge to carry out LES using FastEddy. 

Software and computing requirements
===================================

Computing resources with at least four general purpose graphics processing units are recommended to carry out the test cases. System must be enabled with python and Jupyter notebook packages. Add other requirements (compilers, libraries, etc).

Instructions on how to build and run FastEddy on NCAR's Casper architecture https://github.com/NCAR/FastEddy-model/blob/main/README.md.
