*************************
Post-processing Utilities
*************************

FastEddy provides Python utilities for converting efficient binary output to
NetCDF and for calculating time-averaged statistics from virtual-tower output.
The utilities are located in **scripts/python_utilities/post-processing/**.

The post-processing workflow is:

.. code-block:: text

   FastEddy simulation
          |
          +--> full-domain binary output
          |       |
          |       +--> FEbinaryToNetCDF.py
          |              |
          |              +--> NetCDF files
          |
          +--> virtual-tower binary output
                  |
                  +--> FEtowersToNetCDF.py
                         |
                         +--> combined tower NetCDF
                                  |
                                  +--> TowerTimeStatistics.py
                                         |
                                         +--> time-averaged tower statistics

.. note::

   **FEbinaryToNetCDF.py** and **FEtowersToNetCDF.py** convert the efficient
   binary output produced by FastEddy. **TowerTimeStatistics.py** operates on
   the combined tower NetCDF file, normally produced by
   **FEtowersToNetCDF.py**.

.. contents::
   :local:
   :depth: 2


Full-Domain Binary Conversion
=============================

**FEbinaryToNetCDF.py** converts rank-wise FastEddy full-domain binary output
to one NetCDF file per requested output timestep. The conversion is parallel
over the requested timesteps using MPI.

This is the post-processing utility associated with :code:`ioOutputMode = 1`.
See :doc:`efficient_output` for an overview of FastEddy efficient output modes.

Command Line
------------

Run the converter with:

.. code-block:: console

   mpirun -np <N> python scripts/python_utilities/post-processing/FEbinaryToNetCDF.py \
       -f convert.json \
       -a field_attributes.json

The two required arguments are:

**-f, --file**
   JSON file containing conversion parameters.

**-a, --attrs**
   JSON file containing field metadata used to add units, long names, and
   CF-style standard names to the resulting NetCDF variables.

Configuration: convert.json
---------------------------

The supplied example is:

.. code-block:: json

   {
     "outpath": "INSERT_PATH_TO_YOUR_RUN_DIRECTORY/output_binary/",
     "FEoutBase": "FE_DISPERSION",
     "numOutRanks": 4,
     "fileSetSize": 12,
     "tstart": 0,
     "tstep": 30000,
     "netCDFpath": "INSERT_PATH_TO_YOUR_RUN_DIRECTORY/output_binary/NetCDF/",
     "removeBinaries": true
   }

The parameters are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - **outpath**
     - Directory containing the rank-wise binary files.
   * - **FEoutBase**
     - Base name used in the binary and NetCDF filenames.
   * - **numOutRanks**
     - Number of MPI ranks/GPU output partitions represented in each timestep's
       binary files. This should match the number of rank-wise files produced
       by the simulation.
   * - **fileSetSize**
     - Number of timestep intervals in the conversion batch. The converter
       processes the timestep sequence from **tstart** through
       **tstart + tstep \* fileSetSize**.
   * - **tstart**
     - First simulation timestep to convert.
   * - **tstep**
     - Increment between output timesteps to convert.
   * - **netCDFpath**
     - Directory in which converted NetCDF files are written. It is created if
       necessary.
   * - **removeBinaries**
     - If **true**, the rank-wise binary files for a successfully converted
       timestep are removed after the NetCDF file has been written.

Input and Output
----------------

For a timestep **T**, the converter expects files of the form:

.. code-block:: text

   <outpath>/<FEoutBase>_rank_0.T
   <outpath>/<FEoutBase>_rank_1.T
   ...
   <outpath>/<FEoutBase>_rank_<numOutRanks-1>.T

If all rank files for a timestep are present, they are read, spatially
stitched, annotated with metadata, and written as:

.. code-block:: text

   <netCDFpath>/<FEoutBase>.T

If one or more rank files are missing, that timestep is skipped.

The converter also creates explicit **xIndex**, **yIndex**, and **zIndex**
coordinates when applicable and assigns metadata from
**field_attributes.json**. The metadata lookup handles standard FastEddy
fields, coordinate variables, Jacobian terms, base-state fields, auxiliary
scalars, and moisture-flux fields.

.. important::

   **fileSetSize** controls a batch of output timesteps for one invocation.
   When running on multiple MPI processes, the number of MPI processes must be
   compatible with the requested batch size. The converter checks this
   requirement at startup.


Virtual-Tower Binary Conversion
===============================

**FEtowersToNetCDF.py** combines the per-tower binary files produced by
FastEddy's virtual-tower output into a single time-series NetCDF file.

Virtual towers are enabled with:

.. code-block:: none

   towerIOSelector = 1
   towerPath = ./TowerData/
   towerSpecsFile = ./towerSpecsFile.nc

See :doc:`efficient_output` for the tower-output configuration and
**towerSpecsFile** coordinate conventions.

Command Line
------------

Run:

.. code-block:: console

   python scripts/python_utilities/post-processing/FEtowersToNetCDF.py \
       -f towers.json

The **-f, --file** argument is required and specifies the JSON configuration
file.

Configuration: towers.json
--------------------------

The supplied example is:

.. code-block:: json

   {
     "runPath": "INSERT_PATH_TO_YOUR_RUN_DIRECTORY/",
     "FEparamsFile": "FE_params.in",
     "outputFileName": "Alltowers.nc",
     "startStep": 0,
     "endStep": 300000
   }

The parameters are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - **runPath**
     - Root directory of the FastEddy run. The tower binary directory is
       obtained from the **towerPath** parameter in the FastEddy parameter
       file.
   * - **FEparamsFile**
     - FastEddy parameter file. The converter reads **NtBatch**, **Nz**,
       **towerPath**, **towerSpecsFile**, and output-selector parameters from
       this file to determine the binary layout.
   * - **outputFileName**
     - Name of the combined tower NetCDF file.
   * - **startStep**
     - First simulation timestep to include.
   * - **endStep**
     - Last simulation timestep to include.

The converter can find **FEparamsFile** either beneath **runPath** or in the
current working directory. Likewise, it accepts the tower specification file
where it is found under the run directory or at the supplied path.

Input Files
-----------

For each tower, the converter reads the initial/static file and subsequent
time-series batches. The expected files include:

.. code-block:: text

   tower_ic_<tower>.<batch>
   tower_<tower>.<batch>
   tower_sv_<tower>.<batch>

The exact set of profile variables depends on the FastEddy output selectors
in the parameter file. The converter accounts for optional TKE, moisture,
auxiliary-scalar, and subgrid-flux fields.

Output
------

The resulting NetCDF file contains a **time** dimension plus tower and
vertical dimensions. Profile variables use dimensions:

.. code-block:: text

   (towerID, time, zIndex)

Surface variables use:

.. code-block:: text

   (towerID, time)

The file also includes tower coordinates and metadata such as **z**, **x**,
**y**, **elevation**, **SeaMask**, **xOffset**, and **yOffset**.

The converter converts the pressure-weighted/profile quantities that are
stored in the binary tower output to the corresponding physical quantities
where required by the current FastEddy tower-output convention.


Tower Time Statistics
=====================

**TowerTimeStatistics.py** computes block-averaged statistics from a combined
tower NetCDF file. It is intended for high-frequency virtual-tower analysis
after **FEtowersToNetCDF.py** has produced a time-series NetCDF file.

The script computes, for each tower and vertical level:

* mean **u**, **v**, **w**, and **theta**;
* variances of **u**, **v**, **w**, and **theta**;
* turbulent covariances **u'w'**, **v'w'**, and **theta'w'**;
* mean density **rho** and water-vapor mixing ratio **qv**;
* turbulent moisture flux **qv'w'**;
* mean subgrid TKE (**TKE_0**);
* mean virtual temperature **T_v**;
* mean dry temperature **T_d**;
* mean virtual potential temperature **theta_v**; and
* virtual-temperature turbulent flux **theta_v'w'**.

The output variable names and units are fixed by the implementation:

.. list-table::
   :header-rows: 1
   :widths: 24 18 58

   * - Variable
     - Units
     - Description
   * - **u**, **v**, **w**
     - m s-1
     - Block mean wind components.
   * - **theta**
     - K
     - Block mean potential temperature.
   * - **var_u**, **var_v**, **var_w**
     - m2 s-2
     - Variances of the wind components.
   * - **var_th**
     - K2
     - Potential-temperature variance.
   * - **cov_uw**, **cov_vw**
     - m2 s-2
     - Turbulent momentum covariances.
   * - **cov_thw**
     - K m s-1
     - Potential-temperature turbulent flux.
   * - **rho**
     - kg m-3
     - Block mean air density.
   * - **qv**
     - g kg-1
     - Block mean water-vapor mixing ratio.
   * - **cov_qvw**
     - g kg-1 m s-1
     - Water-vapor turbulent flux.
   * - **tke_0**
     - m2 s-2
     - Block mean grid-filter-scale subgrid TKE.
   * - **T_v**
     - C
     - Block mean virtual temperature.
   * - **T_d**
     - C
     - Block mean dry temperature.
   * - **theta_v**
     - K
     - Block mean virtual potential temperature.
   * - **cov_thvw**
     - K m s-1
     - Virtual-potential-temperature turbulent flux.

Command Line
------------

Run the script with:

.. code-block:: console

   python scripts/python_utilities/post-processing/TowerTimeStatistics.py \
       -f tower_time_stats.json

If **-f** is omitted, the script looks for **tower_time_stats.json** in the
current working directory.

Configuration: tower_time_stats.json
------------------------------------

The supplied example is:

.. code-block:: json

   {
     "input_file": "/path/to/Alltowers.nc",
     "output_path": "/path/to/output",
     "output_name_tag": "_5mNest",
     "case_tag": "",
     "start_datetime_utc": "2025-06-04_12:00:00",
     "time_avg_window_sec": 1800.0,
     "detrend_opt": true,
     "tower_names": ["M1", "S10", "S13", "S14"],
     "overwrite_existing": true,
     "log_level": "INFO"
   }

The parameters are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - **input_file**
     - Path to the combined tower NetCDF input file.
   * - **output_path**
     - Directory in which the statistics NetCDF file is written. It is
       created if necessary.
   * - **output_name_tag**
     - Optional string inserted into the output filename.
   * - **case_tag**
     - Optional case identifier appended to the output filename.
   * - **start_datetime_utc**
     - UTC date and time corresponding to the beginning of the input tower
       record, using **YYYY-MM-DD_HH:MM:SS**. It is used to construct absolute
       UTC timestamps and the output date tag.
   * - **time_avg_window_sec**
     - Averaging interval in seconds. The input time series is divided into
       non-overlapping blocks of approximately this duration based on the
       input sampling interval. Only complete blocks are processed.
   * - **detrend_opt**
     - If **true**, a linear trend is removed from each block before
       calculating second moments and turbulent covariances. If **false**,
       fluctuations are formed by subtracting the block mean.
   * - **tower_names**
     - Optional list of tower names. If provided, its length must equal the
       number of towers in the input file. If omitted or **null**, the tower
       identifiers in the input file are used.
   * - **overwrite_existing**
     - Controls whether an existing output file may be replaced. The default
       is **true**.
   * - **log_level**
     - Python logging level, such as **INFO** or **DEBUG**. The default is
       **INFO**.

.. important::

   **start_datetime_utc** describes the start of the tower record; it is not
   inferred from the NetCDF **time** coordinate. Make sure it matches the
   simulation start time represented by the input file.

Averaging and Detrending
------------------------

The script determines the FastEddy sampling interval from the first two
values of the input **time** variable. The requested averaging window is
converted to a number of samples, and the input is trimmed to a whole number
of averaging blocks.

When **detrend_opt = true**, the script linearly detrends each block before
forming the variances and covariances. NaN-containing fields use a
NaN-aware vectorized detrending implementation. Fields without NaNs use a
faster vectorized implementation.

When **detrend_opt = false**, variances are calculated directly within each
block and perturbations for the covariance calculations are obtained by
subtracting the block mean.

.. note::

   The mean fields are block averages in both modes. The **detrend_opt**
   setting affects the calculation of second moments and covariances, not
   the reported means.

Output File
-----------

The output contains:

.. code-block:: text

   towers_aver_FE(n_tower, var_fe, time, zIndex)
   towers_z(n_tower, zIndex)
   towers_topo(n_tower)
   towerID(n_tower)
   yyyymmdd_FE
   secAver
   time_UTC_sec_day(time)
   time_UTC(time)
   vars_FE(var_fe)
   units_FE(var_fe)

The **vars_FE** and **units_FE** arrays describe the entries in the
**var_fe** dimension.

The output filename is generated as:

.. code-block:: text

   TowerAver_Nz<Nz><output_name_tag>_<window_seconds><detrend>_<YYYYMMDD><case_tag>.nc

where **<detrend>** is **sDetrended** when detrending is enabled and **s**
otherwise.

For example, a 1800-second, detrended calculation beginning on
2025-06-04 with four towers and **output_name_tag = "_5mNest"** produces a
filename similar to:

.. code-block:: text

   TowerAver_Nz<Nz>_5mNest_1800sDetrended_20250604.nc

Dependencies
------------

The utilities require a Python environment with the packages needed by the
individual scripts:

* **numpy**
* **xarray**
* **pandas** (used by the binary conversion utilities)
* **mpi4py** (required by **FEbinaryToNetCDF.py**)

The resulting NetCDF files require a NetCDF4-capable backend for the
full-domain converter.

Metadata lookup: field_attributes.json
--------------------------------------

**field_attributes.json** is used by **FEbinaryToNetCDF.py** to attach
metadata to converted full-domain NetCDF variables. It is a lookup table,
not a user-facing command-line configuration file.

It contains groups for:

* base fields;
* Jacobian/metric terms;
* coordinate variables;
* direction names; and
* special base-state field mappings.

When new output fields are added to FastEddy, this file should be updated if
the field needs explicit units, a long name, or a CF standard name in the
converted NetCDF output.


Recommended Workflow
====================

For full-domain efficient output:

.. code-block:: console

   mpirun -np <N> python scripts/python_utilities/post-processing/FEbinaryToNetCDF.py \
       -f convert.json \
       -a field_attributes.json

For virtual towers:

.. code-block:: console

   python scripts/python_utilities/post-processing/FEtowersToNetCDF.py \
       -f towers.json

   python scripts/python_utilities/post-processing/TowerTimeStatistics.py \
       -f tower_time_stats.json

The second command uses the combined tower NetCDF generated by the first
tower-conversion command. In practice, it is usually best to keep the
converter output and the statistics output in separate directories so that
raw conversion products are not accidentally overwritten.

