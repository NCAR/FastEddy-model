**********************
Efficient Output Modes
**********************

.. |br| raw:: html

   <br/>

When running FastEddy\ :sup:`®` simulations, there are two options available for efficient model output (full domain and vertical profiles). In both cases, data is written in binary format, and Python scripts are provided to facilitate conversion to netCDF format.

Full domain
===========
For cases where large domains are considered and full domain output is required, an efficient performance alternative to the standard N-to-1 gather and write of netCDF outputs is available. This option is activated with the following parameter in the FastEddy parameters file:

.. code-block:: none

   #--IO
   ioOutputMode = 1

In this case, each MPI-rank will write binary output filesi corresponding to the rank-wise subdomain. A user may then personalize and execute the batch submission script **/scripts/batch_jobs/fasteddy_convert_pbs_script_casper.sh** which will invoke a python script (**/scripts/python_utilities/post-processing/FEbinaryToNetCDF.py**) to convert the rank-wise binary files from each output timestep into a single aggregate netCDF output file per timestep. Users can run the following `conda activate` command if running on Casper:

.. code::

   conda activate /glade/u/fehelp/casper/conda-envs/mpi4py-casper-oneapi-2024.2.1-openmpi-5.0.6

The **convert.json** file controls the specifics of the conversion as follows:

.. csv-table::
   :file: csv/efficient_output_full.csv
   :header-rows: 1
   :delim: ;
   :class: binary
   :widths: 18, 82

.. note::

  - FastEddy can only restart from a netCDF file, irrespective of the :code:`ioOutputMode` options used.

Profiles
========
An efficient option to output high temporal fidelity vertical profile (spanning the full vertical extent) time-series (every model timestep) at specific locations is available. This option is activated with the following selector in the FastEddy parameters file:

.. code-block:: none

   #--IO
   towerIOSelector = 1
   towerPath = ./TowerData/
   towerSpecsFile = ./towerSpecsFile.nc

*towerPath* indicates the location where the profiles will be written. A netCDF file (*towerSpecsFile*) specifies the location of the vertical profiles in either lat,lon coordinates for real-case domains, or in cartesian x,y coordinates of a targeted real or idealized FastEddy domain. For lat,lon profile (virtual tower) coordinates this file should have following structure:

.. code-block:: none

   int coordType ;
   float coordsLat(nProfs) ;
   float coordsLon(nProfs) ;

where :code:`coordType` = 0 is used when the coordinates are specified based on latitude (:code:`coordsLat`), and longitude (:code:`coordsLon`). Alternatively, the cartesian x,y coordinates refered to the FastEddy domain grid can be specified. In this case set :code:`coordType` = 1 and provide coordinates variable arrays :code:`coordsSN` and :code:`coordsWE`.

.. code-block:: none

   int coordType ;
   float coordsSN(nProfs) ;
   float coordsWE(nProfs) ;

A python script (**/scripts/python_utilities/post-processing/FEtowersToNetCDF.py**) to convert the profile binary files into a single aggregate netCDF output file is provided. The **towers.json** file controls the specifics of the conversion as follows:

.. csv-table::
   :file: csv/efficient_output_prof.csv
   :header-rows: 1
   :delim: ;
   :class: binary
   :widths: 18, 82
