=================================================
Boundary layer in the presence of resolved canopy
=================================================

Background
----------

This is an idealized scenario of a neutrally stratified boundary layer in the presence of a horizontally homogeneous canopy. This case utilizes a two-equation canopy model (Shaw and Scumann, 1992; Shaw and Patton, 2003).

Input parameters
----------------

* Number of grid points: :math:`[N_x,N_y,N_z]=[252,250,90]`
* Isotropic grid spacings in the horizontal directions: :math:`[dx,dy]=[4,4]` m, vertical grid is :math:`dz=4` m at the surface and stretched with verticalDeformFactor :math:`=0.25`
* Domain size: :math:`[1.0 \times 1.0 \times 1.44]` km
* Model time step: :math:`0.04` s
* Advection scheme: 5th-order upwind
* Time scheme: 3rd-order Runge Kutta
* Geostrophic wind: :math:`[U_g,V_g]=[10,0]` m/s
* Latitude: :math:`54.0^{\circ}` N
* Surface potential temperature: :math:`300` K
* Potential temperature profile:

.. math::
  \partial{\theta}/\partial z =
    \begin{cases}
      0 & \text{if $z$ $\le$ 500 m}\\
      0.08 & \text{if 500 m < $z$ $\le$ 650 m}\\
      0.003 & \text{if $z$ > 650 m}
    \end{cases} 

* Surface heat flux:  :math:`0.0` Km/s
* Surface roughness length: :math:`z_0=1e-12` m
* Rayleigh damping layer: uppermost :math:`600` m of the domain
* Initial perturbations: :math:`\pm 0.25` K 
* Depth of perturbations: :math:`375` m
* Top boundary condition: free slip
* Lateral boundary conditions: periodic
* Time period: :math:`4` h

Execute FastEddy
----------------

Run FastEddy using the input parameters file /examples/Example01_NBL.in. To execute FastEddy, follow the instructions here: https://github.com/NCAR/FastEddy-model/blob/main/README.md.

Visualize the output
--------------------

Open the Jupyter notebook entitled "MAKE_FE_TUTORIAL_PLOTS.ipynb" and execute it using setting: case = 'neutral'.

XY-plane views of instantaneous velocity components at :math:`t=7` h (FE_NBL.630000):

.. image:: ../images/UVWTHETA-XY-neutral.png
  :width: 1200
  :alt: Alternative text
  
XZ-plane views of instantaneous velocity components at :math:`t=7` h (FE_NBL.630000):

.. image:: ../images/UVWTHETA-XZ-neutral.png
  :width: 900
  :alt: Alternative text
  
Mean (domain horizontal average) vertical profiles of state variables at :math:`t=7` h (FE_NBL.630000):

.. image:: ../images/MEAN-PROF-neutral.png
  :width: 750
  :alt: Alternative text
 
Horizontally-averaged vertical profiles of turbulence quantities at :math:`t=6-7` h [perturbations are computed at each time instance from horizontal-slab means, then averaged horitontally and over the previous 1-hour mean]:

.. image:: ../images/TURB-PROF-neutral.png
  :width: 1200
  :alt: Alternative text 

Analyze the output
------------------

* Using the XY and XZ cross sections, discuss the characteristics (scale and magnitude) of the resolved turbulence.
* What is the boundary layer height in the neutral case?
* Using the vertical profile plots, explain why the boundary layer is neutral.

