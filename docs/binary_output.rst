**********************
Efficient Output Modes
**********************

.. |br| raw:: html

   <br/>

When running FastEddy\ :sup:`®` simulations, there are two options available for efficient model output (full domain and vertical profiles). In both cases, data is written in binary format, and Python scripts are provided to facilitate conversion to netCDF format.

Full domain
===========
For cases where large domains are considered and/or full domain output is required, an efficient alternative to the standard netCDF output is provided. This option is activated with the following parameter in the FastEddy parameters file:

.. code-block:: none

   #--IO
   ioOutputMode = 1 # 0: N-to-1 gather and write to a netcdf file, 1:N-to-N writes of FastEddy binary files

In this case, rank-wise binary output files. Personalize and use the batch submission script **/scripts/batch_jobs/fasteddy_convert_pbs_script_casper.sh** which will invoke a python script (**/scripts/python_utilities/post-processing/FEbinaryToNetCDF.py**) to convert the rank-wise binary files from each output timestep into a single aggregate netCDF output file per timestep. Users can run the following `conda activate` command if running on Casper:

.. code::

   conda activate /glade/u/fehelp/casper/conda-envs/mpi4py-casper-oneapi-2024.2.1-openmpi-5.0.6

The **convert.json** file controls the specifics of the conversion as follows:

.. csv-table::
   :file: csv/efficient_output.csv
   :header-rows: 1
   :delim: ;
   :class: efficientoutput


