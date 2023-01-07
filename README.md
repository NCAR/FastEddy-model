# FastEddy® 
©2016 University Corporation for Atmospheric Research

# Open-source License 
The FastEddy® v1.0 beta release is licensed under the Apache License, Version 2.0 (the "License");
you may not use any source code in this repository except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# Description
FastEddy® (FE) is a large-eddy simulation (LES) model developed by the Research Applications Laboratory at the National Center for Atmospheric Research in Boulder, Colorado USA. The cornerstone premise of FastEddy model development is to leverage the accelerated and more power efficient computing capacity of graphics processing units (GPU)s to exploit the inherent fine-grained parallelism of LES to enable the adoption of microscale and multiscale atmospheric boundary layer modeling.

# Beta-build
The Makefile-based build system included here assumes deployment on the NCAR Casper system https://arc.ucar.edu/knowledge_base/70549550. FastEddy requires a C-compiler, MPI, and CUDA. On Casper ensure modules are loaded for openmpi, netcdf, and cuda with module -t list, and e.g. module load [intel or gnu/openmpi/cuda] as necessary. 
1. Navigate to SRC/FEMAIN
2. To build the FastEddy executable run make (optionally run make clean first if appropriate).

To build on other HPC systems with NVIDIA GPUs, check for availability of the aformentioned modules/dependencies. Successful compilation may require modifications to shell environment variable include or library paths, or alternatively minor adjustments to the include or library flags in SRC/FEMAIN/Makefile.    

# Getting Started
FastEddy Tutorials for idealized cases can be found at https://fasteddytutorial.readthedocs.io

# References
Model Publications: 
1. FastEddy dry dynamics formulation, idealized case validation and performance benchmarks: https://doi.org/10.1029/2020MS002100
2. FastEddy moist dynamics extension and validation: https://doi.org/10.1029/2021MS002904  
