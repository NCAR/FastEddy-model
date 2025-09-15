**********
Parameters
**********

This page defines the configurable parameters available in FastEddy\ :sup:`®`.
Parameters are organized into <fill in>.

Each table provides the parameter name,
valid range where applicable, requirement status (required or optional), and a
brief description. These definitions serve as a reference to ensure correct
configuration and valid inputs for FastEddy simulations.

**FORMAT OPTION 1**

.. csv-table:: MPI AALES
   :file: csv/mpi_aales.csv
   :header-rows: 1
   :delim: ,
   :widths: 30, 18, 18, 20, 35
   :class: longtable
	   
.. csv-table:: CUDA AALES
   :file: csv/cuda_aales.csv
   :header-rows: 1
   :delim: ,
   :widths: 30, 18, 18, 20, 35
   :class: longtable
	     
.. csv-table:: IO
   :file: csv/io.csv
   :header-rows: 1
   :delim: ,
   :widths: 30, 18, 18, 20, 35
   :class: longtable

.. csv-table:: GRID
   :file: csv/grid.csv
   :header-rows: 1
   :delim: ,
   :widths: 30, 18, 18, 20, 35
   :class: longtable

.. csv-table:: TIME INTEGRATION
   :file: csv/time_integration.csv
   :header-rows: 1
   :delim: ,
   :widths: 30, 18, 18, 20, 35
   :class: longtable

**FORMAT OPTION 2**

+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| **Name**                | **Min**   | **Max**   | **Mandatory**          | **Description**                               |
|                         |           |           | **or Optional**        |                                               |
+=========================+===========+===========+========================+===============================================+
| **MPI AALES**           |           |           |                        |                                               |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| numProcsX               | 1         | INT_MAX   | Mandatory              | Number of cores to be used for horizontal     |
|                         |           |           |                        | domain decomposition in X                     |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| numProcsY               | 1         | INT_MAX   | Mandatory              | Number of cores to be used for horizontal     |
|                         |           |           |                        | domain decomposition in Y                     |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| **CUDA AALES**          |           |           |                        |                                               |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| tBx                     | 1         | INT_MAX   | Mandatory              | Number of threads in x-dimension              |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| tBy                     | 1         | INT_MAX   | Mandatory              | Number of threads in y-dimension              |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| tBz                     | 1         | INT_MAX   | Mandatory              | Number of threads in z-dimension              |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| **IO**                  |           |           |                        |                                               |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| inPath                  | N/A       | N/A       | Optional               | Path where initial/restart file is read in    |
|                         |           |           |                        | from                                          |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| inFile                  | N/A       | N/A       | Optional               | Name of the input file for coordinate system  |
|                         |           |           |                        | and initial or restart conditions             |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| outPath                 | N/A       | N/A       | Mandatory              | Path where output files are to be written     |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| outFileBase             | N/A       | N/A       | Mandatory              | Base name of the output file series as in     |
|                         |           |           |                        | (outFileBase).element-in-series               |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| frqOutput               | 0         | INT_MAX   | Mandatory              | Frequency (in timesteps) at which to produce  |
|                         |           |           |                        | output                                        |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| ioOutputMode            | 0         | 1         | Optional               | 0: N-to-1 gather and write to a netcdf file;  |
|                         |           |           |                        | 1:N-to-N writes of FastEddy binary files      |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| **GRID**                |           |           |                        |                                               |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| Nx                      | 1         | INT_MAX   | Mandatory              | Number of discretised domain elements in the  |
|                         |           |           |                        | x (zonal) direction                           |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| Ny                      | 1         | INT_MAX   | Mandatory              | Number of discretised domain elements in the  |
|                         |           |           |                        | y (meridional) direction                      |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| Nz                      | 1         | INT_MAX   | Mandatory              | Number of discretised domain elements in the  |
|                         |           |           |                        | z (vertical) direction                        |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| Nh                      | 0         | INT_MAX   | Mandatory              | Number of halo cells to be used (dependent    |
|                         |           |           |                        | on largest stencil extent)                    |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| d_xi                    | FLT_MIN   | FLT_MAX   | Mandatory              | Computational domain fixed resolution in the  |
|                         |           |           |                        | 'i' direction                                 |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| d_eta                   | FLT_MIN   | FLT_MAX   | Mandatory              | Computational domain fixed resolution in the  |
|                         |           |           |                        | 'j' direction                                 |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| d_zeta                  | FLT_MIN   | FLT_MAX   | Mandatory              | Computational domain fixed resolution in the  |
|                         |           |           |                        | 'k' direction                                 |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| coordHorizHalos         | 0         | 1         | Mandatory              | Switch to setup coordiante halos as           |
|                         |           |           |                        | periodic=1 or gradient-following=0            |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| topoFile                | N/A       | N/A       | Optional               | A file containing topography (surface         |
|                         |           |           |                        | elevation in meters ASL)                      |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| verticalDeformSwitch    | 0         | 1         | Mandatory              | Switch to use vertical coordinate deformation |
|                         |           |           |                        | 0=off, 1=on                                   |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| verticalDeformFactor    | 0.0       | 1.0       | Mandatory              | Deformation factor (0.0=max compression;      |
|                         |           |           |                        | 1.0=no compression)                           |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| verticalDeformQuadCoeff | -2.0      | 2.0       | Mandatory              | Deformation factor (0.0=max compression;      |
|                         |           |           |                        | 1.0=no compression)                           |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| **TIME INTEGRATION**    |           |           |                        |                                               |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| timeMethod              | 0         | 0         | Mandatory              | Selector for time integration method.         |
|                         |           |           |                        | [0=RK3-WS2002 (default)]                      |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| Nt                      | 1         | INT_MAX   | Mandatory              | Number of timesteps to perform                |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| dt                      | FLT_MIN   | FLT_MAX   | Mandatory              | Timestep resolution in seconds                |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+
| NtBatch                 | 1         | Nt        | Mandatory              | Number of timesteps to compute in batch       |
|                         |           |           |                        | launch; must have NtBatch <= Nt               |
+-------------------------+-----------+-----------+------------------------+-----------------------------------------------+




