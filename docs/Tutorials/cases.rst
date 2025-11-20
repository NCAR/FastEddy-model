***********
Ideal Cases
***********

Seven idealized cases are provided:

* Dry neutral boundary layer
* Dry convective boundary layer
* Dry stable boundary layer
* Moist cloud-topped boundary layer
* Canopy inclusive boundary layer
* Offshore boundary layer
* Passive scalar transport and dispersion over an idealized hill

Resources required for tutorial cases including python utilities and Jupyter Notebooks are provided in the tutorials directory of the `FastEddy-model GitHub repository <https://github.com/NCAR/FastEddy-model>`_ with required data for the moist dynamics example available at this `Zenodo record <https://zenodo.org/records/10982246>`_. All cases are idealized setups over flat terrain. For each case, the user will set up the input parameter file, execute FastEddy, visualize the output using a Jupyter notebook, and perform some basic analysis of the output. After examining the cases, the user will carry out some sensitivity tests by changing various input parameters. The purpose of these tests are for the user to become more familiar with the input parameters, and how changes to those parameters affect the output. After completing these tutorials, the user will have some basic knowledge and capacity to carry out LES activites using FastEddy.

.. toctree::

   cases/NBL.rst
   cases/CBL.rst
   cases/SBL.rst
   cases/MBL.rst
   cases/CANOPY.rst
   cases/OFFSHORE.rst
   cases/DISPERSION.rst

.. rubric:: Sensitivity Tests

* Re-run the neutral case with :math:`[N_x,N_y,N_z]=[400,400,122]` and isotropic grid spacings of :math:`[dx,dy,dz]=[10,10,10]`. Adjust the model time step accordingly. Re-make all plots and discuss the differences between the control case. How much longer did it take to complete the simulation?
* Re-run the convective case with a surface heat flux of :math:`=+0.70` Km/s. Re-make all plots and discuss the differences between the control case.
* Re-run the neutral case with :math:`z_0=0.3` m. Re-make all plots and discuss the differences between the control case.
* Re-run the neutral case with the first order upwind advection scheme. Re-make all plots and discuss the differences between the control case. Why is the first order scheme a bad choice?
* Re-run the stable case with a surface cooling rate of :math:`-0.5` K/h. Re-make all plots and discuss the differences between the control case.
* Re-run the stable case using half of the GPUs used in the control simulation. How much slower does the case run?

* Re-run the BOMEX case with a higher-order advection for water vapor (moistureAdvSelectorQv = 3). What is the impact of the increased effective resolution on dynamical, thermodynamical and microphysical quantities, along with turbulence variability and fluxes? How does that change influce the comparison to the other BOMEX LES models?

.. only

    (Here, the user will make some modifications to the default parameters such as changing the grid spacing, stretching, model time step, advection     

    scheme,    number of grid points, domain decomposition and number of GPUs, etc, etc. Here, the user will execute the sensitivity test, and visualize 
    and analyze the     output)

     x^2+y^2=z^2

      frac{ sum_{t=0}^{N}f(t,k) }{N}

