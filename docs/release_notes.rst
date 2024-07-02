*************
Release Notes
*************

When applicable, release notes are followed by the GitHub issue number which
describes the bugfix, enhancement, or new feature
(`FastEddy-model GitHub issues <https://github.com/NCAR/FastEddy-model/issues>`_).

FastEddy-model Version 1.1 Release Notes (20240422)
===================================================

This is the initial release of the FastEddy, an NSF NCAR developed parallelized
and GPU-resident, large-eddy simulation code for accelerated modeling of the
atmospheric boundary layer.

In addition to the initial code, this release includes a patch for building
the system in the current NSF NCAR high performance computing environment on the
Casper and Derecho platforms, along with other changes as detailed below:

  .. dropdown:: Repository, build, and test

     * Add templates for Issues and Pull Requests (`#1 <https://github.com/NCAR/FastEddy-model/issues/1>`_)
     * Set up the FastEddy Tutorial documentation (`#3 <https://github.com/NCAR/FastEddy-model/issues/3>`_)
     * Consolidate FastEddy-tutorials content into FastEddy-model (`#8 <https://github.com/NCAR/FastEddy-model/issues/8>`_)
     * Adjust FastEddy-tutorials BOMEX notebook & RTD Moist dynamics instructions for hosting datasets under new repo (`#10 <https://github.com/NCAR/FastEddy-model/issues/10>`_)  

  .. dropdown:: Bugfixes

     * Fix to the restart model capability (`#6 <https://github.com/NCAR/FastEddy-model/pull/6>`_)
     * Clean compile with warnings addressed (`#6 <https://github.com/NCAR/FastEddy-model/pull/6>`_)  

  .. dropdown:: Enhancements

     * Accommodate building on Derecho (`#6 <https://github.com/NCAR/FastEddy-model/pull/6>`_)


