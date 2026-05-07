*****************************************
Pre-Processing Parameters Reference Guide
*****************************************

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


.. figure:: /_static/FastEddy_Pre-Processing_Workflow_Gemini.png
   :alt: FastEddy Pre-Processing Workflow Diagram
   :align: center
   :width: 100%

   FastEddy Pre-Processing Workflow Diagram
   
   *Source: Google. (2026). Gemini (Version 3 Flash) [Large language model]. https://gemini.google.com*


geospec.json
============

.. csv-table::
   :file: csv/geospec.csv
   :header-rows: 1
   :delim: ;
   :class: preprocessing


simgrid.json
============

.. csv-table::
   :file: csv/simgrid.csv
   :header-rows: 1
   :delim: ;
   :class: preprocessing

genicbcs.json
=============

.. csv-table::
   :file: csv/genicbcs_top.csv
   :header-rows: 1
   :delim: ;
   :class: preprocessing

WRF as Parent Domain Section
----------------------------

Active only when parent_model=0.

.. csv-table::
   :file: csv/genicbcs_wrf.csv
   :header-rows: 1
   :delim: ;
   :class: preprocessing


FastEddy as Parent Domain Section
---------------------------------

Active only when parent_model = 1.

.. csv-table::
   :file: csv/genicbcs_fe.csv
   :header-rows: 1
   :delim: ;
   :class: preprocessing
