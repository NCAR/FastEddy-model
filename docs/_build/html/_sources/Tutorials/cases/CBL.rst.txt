=============================
Dry convective boundary layer
=============================

This is the convective boundary layer scenario described by Sauer and Munoz-Esparza (2020). This case represents the boundary layer conditions at the SWiFT facility near Lubbock, Texas at 4 July 2012 during the period of 18Z-20Z (12:00–14:00 local time), the strongest period of convection on the day.

Input parameters
----------------

* Number of grid points: :math:`[N_x,N_y,N_z]=[600,594,122]`
* Isotropic grid spacings in the horizontal directions: :math:`[dx,dy]=[20,20]` m, vertical grid is :math:`dz=20` m at the surface and stretched with verticalDeformFactor :math:`=0.80`
* Domain size: :math:`[12.0 \times 11.9 \times 3.0]` km
* Model time step: :math:`0.05` s
* Geostrophic wind: :math:`[U_g,V_g]=[9,0]` m/s
* Advection scheme: Hybrid 5th order upwind
* Time scheme: 3rd-order Runge Kutta
* Latitude: :math:`33.5^{\circ}` N
* Surface potential temperature: :math:`309` K
* Potential temperature profile:

.. math::
  \partial{\theta}/\partial z =
    \begin{cases}
      0 & \text{if $z$ $\le$ 600 m}\\
      0.004 & \text{if $z$ > 600 m}
    \end{cases}

* Surface heat flux:  :math:`0.35` Km/s
* Surface roughness length: :math:`z_0=0.05` m
* Rayleigh damping layer: uppermost :math:`400` m of the domain
* Initial perturbations: :math:`\pm 0.25` K 
* Depth of perturbations: :math:`400` m
* Top boundary condition: free slip
* Lateral boundary conditions: periodic
* Time period: :math:`4` h

Execute FastEddy
----------------

1. Create a working directory to run the FastEddy tutorials and change to that directory.
2. Create a **Example02_CBL** subdirectory and change to that directory.
3. The FastEddy code will write its output to an **output** subdirectory. Create an **output** directory, if one does not already exist.   
4. Run FastEddy using the input parameters file *Example02_CBL.in* located in the **tutorials/examples/** subdirectory of the FastEddy repository. 

See :ref:`run_fasteddy` for instructions on how to build and run FastEddy on NSF NCAR's High Performance Computing machines.

Visualize the output
--------------------

1. Open the Jupyter notebook entitled *MAKE_FE_TUTORIAL_PLOTS.ipynb*.
2. Under the "Define parameters" section, modify :code:`path_base`, specifying the full path to the **Example02_CBL** subdirectory, but don't include the **Example02_CBL** subdirectory. Be sure to include a trailing slash :code:`/`).
3. Under the "Define parameters" section, modify :code:`case` to set its value to :code:`convective`.
4. Run the Jupyter notebook.
5. The resulting XY cross section png plots will be placed in a **FIGS** subdirectory of the **Example02_CBL** directory.

XY-plane views of instantaneous velocity components at :math:`t=4` h (FE_CBL.288000):

.. image:: ../images/UVWTHETA-XY-convective.png
  :width: 1200
  :alt: Alternative text
  
XZ-plane views of instantaneous velocity components at :math:`t=4` h (FE_CBL.288000):

.. image:: ../images/UVWTHETA-XZ-convective.png
  :width: 900
  :alt: Alternative text
  
Mean (domain horizontal average) vertical profiles of state variables at :math:`t=4` h (FE_CBL.288000):

.. image:: ../images/MEAN-PROF-convective.png
  :width: 750
  :alt: Alternative text
  
Horizontally-averaged vertical profiles of turbulence quantities :math:`t=3-4` h [perturbations are computed at each point relative to the previous 1-hour mean, and then horizontally averaged]:

.. image:: ../images/TURB-PROF-convective.png
  :width: 1200
  :alt: Alternative text

Analyze the output
------------------

* Using the XY and XZ cross sections, discuss the characteristics (scale and magnitude) of the resolved turbulence.
* What is the boundary layer height in the convective case?
* Using the vertical profile plots, explain why the boundary layer is unstable.
