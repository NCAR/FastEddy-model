==========================================================
Flow in the presence of a turbine array under a wind shift
==========================================================

This is an idealized scenario of wind farm (3 x 3 turbine array) flow in neutrally stratified boundary layer undergoing a change in wind direction of 90 degrees over 40 minutes. This idealized scenario demonstrates the generalized actuator disk (GAD) implementation in FastEddy (*Sanchez Gomez et al., 2024* [#f1]_), with the inclusion of a turbine yawing capability to align with the meteorological wind direction at the turbine's nacelle. The initial and boundary conditions for this idealized case are derived from a horizontally averaged LES run of a neutral ABL with a geostrophic wind aligned in the zonal direction (:math:`[U_g,V_g]=[10.0,0.0]` m/s) with and a latitude of :math:`40.0^{\circ}` N. The required datasets to run this tutorial are provided at this Zenodo record [TO BE UPDATED].

In order to activate the GAD model, the corresponding selector needs to be turned on (:code:`GADSelector = 1`) in the parameters file. The lines below correspond to additions and modifications to the FastEddy parameters file necessary for turbine-inclusive LES runs (corresponding to the test case from **tutorials/examples/Example09_GAD.in**), particularly for the case when the turbine forces are added to the model's output.

.. code-block:: none

   #--GAD
   GADSelector = 1
   turbineSpecsFile = ./GAD_NREL28_9WTs_tutorial.nc
   GADoutputForces = 1

A key aspect required for the GAD model to work is the specification of the aerodynamic characteristics of the simulated turbine. These are included in a netCDF file, together with geometrical characteristics of the wind turbine (rotor diameter: :code:`GAD_rotorD`, turbine hub height: :code:`GAD_hubHeights`, nacelle diameter: :code:`GAD_nacelleD`) and the initial location and orientation of the turbines (:code:`GAD_Xcoords`, :code:`GAD_Ycoords`, :code:`GAD_rotorTheta`). Polynomial fits for lift and drag coefficient, twist, chord length, blade pitch, and rotational speed are utilized, discretized over a finite number of normalized blade elements (:code:`rnorm_vect`), required by the blade-element momentum theory used in the GAD formulation. For flexibilty purposes, an arbitrary number of turbines can be defined (:code:`GAD_turbineType`). This tutorial provides an example turbine specification file corresponding to NREL28's turbine (**GAD_NREL28_9WTs_tutorial.nc**). All the required variables and dimensions in the turbine specifications file are listed here below.
