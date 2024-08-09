****************
Ideal Test Cases
****************

Seven test cases are described:

* Dry neutral boundary layer
* Dry convective boundary layer
* Dry stable boundary layer
* Moist cloud-topped boundary layer
* Canopy inclusive boundary layer
* Offshore boundary layer
* Passive scalar transport and dispersion over an idealized hill

Required tutorial resources including python utilities and Jupyter Notebooks are provided in the tutorials directory of the `FastEddy-model GitHub repository <https://github.com/NCAR/FastEddy-model>`_ with required data for the moist dynamics example available at this `Zenodo record <https://zenodo.org/records/10982246>`_. All test cases are idealized setups over flat terrain. For each case, the user will set up the input parameter file, execute FastEddy, visualize the output using a Jupyter notebook, and perform some basic analysis of the output. After examining the test cases, the user will carry out some sensitivity tests by changing various input parameters. The purpose of these tests are for the user to become more familiar with the input parameters, and how changes to those parameters affect the output. After the tutorial, the user is expected to have basic knowledge to carry out LES using FastEddy.

.. toctree::

   cases/NBL.rst
   cases/CBL.rst
   cases/SBL.rst
   cases/MBL.rst
   cases/CANOPY.rst
   cases/OFFSHORE.rst
   cases/DISPERSION.rst
