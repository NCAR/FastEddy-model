==========================================================
Flow in the presence of a turbine array under a wind shift
==========================================================

This is an idealized scenario of wind farm (3 x 3 turbine array) flow in neutrally stratified boundary layer undergoing a change in wind direction of 90 degrees over 40 minutes. This idealized scenario demonstrates the generalized actuator disk (GAD) implementation in FastEddy (*Sanchez Gomez et al., 2024*), with the inclusion of a turbine yawing capability to align with the meteorological wind direction at the turbine's nacelle. The initial and boundary conditions for this idealized case are derived from a horizontally averaged LES run of a neutral ABL with a geostrophic wind aligned in the zonal direction (:math:`[U_g,V_g]=[10.0,0.0]` m/s) with and a latitude of :math:`40.0^{\circ}` N. The required datasets to run this tutorial are provided at this Zenodo record [TO BE UPDATED].

In order to activate the GAD model, the corresponding selector needs to be turned on (:code:`GADSelector = 1`) in the parameters file. The lines below correspond to additions and modifications to the FastEddy parameters file necessary for turbine-inclusive LES runs (corresponding to the test case from **tutorials/examples/Example09_GAD.in**), particularly for the case when the turbine forces are added to the model's output.

