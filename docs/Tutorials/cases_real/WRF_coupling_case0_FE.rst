========================================================
Running a real-world downscaled simulation with FastEddy
========================================================

After initial and boundary conditions have been properly created, FastEddy can now be ran forced by the mesoscale WRF fileds. The lines below correspond to additions and modifications to the FastEddy parameters file as required for coupled mesoscale-LES runs (corresponding to the test case from **tutorials/examples/Example08_REALCASE_FortCollins.in**)

.. code-block:: none

   #--GRID
   topoFile = ./FortCollinsCO_Topography_448x450.dat
   #--Boundary Conditions Set
   hydroBCs = 1
   ceilingAdvectionBC = 1
   hydroBndysFileBase = ./ICBC/FE_Bndys
   hydroBndysFileStart = 0
   hydroBndysFileEnd = 16
   dtBdyPlaneBCs = 300.0

From the parameters above, :code:`hydroBCs = 1` is the main option to activate the coupling to a mesoscale model. Note that :code:`dtBdyPlaneBCs` is the frequency in seconds for boundary conditions to updte and that it needs to match the value of *secInc* specified in **genicbcs.json**. See Running under NSF NCAR HPC for instructions on how to build and run FastEddy on NSF NCAR’s High Performance Computing machines.
