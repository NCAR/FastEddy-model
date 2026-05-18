******************************
Pre-Processing Reference Guide
******************************

.. |br| raw:: html

   <br/>

This page defines the configurable pre-processing parameters available in
FastEddy\ :sup:`®` and provides a workflow image.

Parameters are organized by the JSON file in which
they appear.

Each table provides the:

  * Name
  * Description

These definitions serve as a reference to ensure correct configuration.

See the :ref:`pre-processing_workflow_diagram` for a visual representation of the workflow.

geospec.json
============

.. csv-table::
   :file: csv/geospec.csv
   :header-rows: 1
   :delim: ;
   :class: preprocessing
   :widths: 22, 78

simgrid.json
============

.. csv-table::
   :file: csv/simgrid.csv
   :header-rows: 1
   :delim: ;
   :class: preprocessing
   :widths: 22,	78

genicbcs.json
=============

.. csv-table::
   :file: csv/genicbcs_top.csv
   :header-rows: 1
   :delim: ;
   :class: preprocessing
   :widths: 22,	78

WRF as Parent Domain Section
----------------------------

Active only when parent_model=0.

.. csv-table::
   :file: csv/genicbcs_wrf.csv
   :header-rows: 1
   :delim: ;
   :class: preprocessing
   :widths: 22,	78


FastEddy as Parent Domain Section
---------------------------------

Active only when parent_model = 1.

.. csv-table::
   :file: csv/genicbcs_fe.csv
   :header-rows: 1
   :delim: ;
   :class: preprocessing
   :widths: 22,	78
