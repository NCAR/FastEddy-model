# FastEddy® 
©2016 University Corporation for Atmospheric Research

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11042754.svg)](https://doi.org/10.5281/zenodo.11042754)

# Open-source License 
The FastEddy® model is licensed under the Apache License, Version 2.0 (the "License");
you may not use any source code in this repository except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# Description
FastEddy® (FE) is a large-eddy simulation (LES) model developed by the Research Applications Laboratory (RAL) at the U.S. National Science Foundation National Center for Atmospheric Research (NSF NCAR) in Boulder, Colorado, USA. The fundamental premise of FastEddy model development is to leverage the accelerated and more power efficient computing capacity of graphics processing units (GPU)s to enable not only more widespread use of LES in research activities but also to pursue the adoption of microscale and multiscale, turbulence-resolving, atmospheric boundary layer modeling into local scale weather prediction or actionable science and engineering applications.

## Contact
Please submit all comments, feedback, suggestions, or questions by email to the NSF NCAR FastEddy team at [fasteddy@ucar.edu](fasteddy@ucar.edu). Further information about FastEddy applications and research is available via the [RAL website](https://ral.ucar.edu/solutions/products/fasteddy). 

## Citation
FastEddy should be cited as follows:

Sauer, J., and D. Muñoz-Esparza. "The FastEddy resident-GPU accelerated large-eddy
  simulation framework: model formulation, dynamical-core validation and performance
  benchmarks". *Journal of Advances in Modeling Earth Systems*, vol. 12 (2020)
  https://doi.org/10.1029/2020MS002100


## Building FastEddy

### Prerequisites

The following dependencies are required to build FastEddy:

| Dependency | Description |
|---|---|
| **CUDA Toolkit** (≥ 11.0) | NVIDIA CUDA compiler (`nvcc`) and runtime libraries |
| **MPI** | Message Passing Interface (e.g., OpenMPI, MPICH) |
| **NetCDF C Library** | NetCDF C interface for I/O operations |
| **CMake** (≥ 3.21) | Build system generator |
| **GNU Make** or **Ninja** | Build tool |

### Build Configuration

FastEddy uses CMake. Configure the build from an out-of-source directory:

```bash
mkdir build && cd build
cmake .. [options]
```

#### CMake Options

| Option | Description | Default |
|---|---|---|
| `-DUSE_HIP=ON/OFF` | Use AMD HIP/ROCm backend instead of NVIDIA CUDA | `OFF` |
| `-DWITH_GAD=ON/OFF` | Enable the GAD (Generalized Actuator Disk) extension module | `OFF` |
| `-DWITH_URBAN=ON/OFF` | Enable the building-resolving URBAN extension module | `OFF` |
| `-DCMAKE_CUDA_ARCHITECTURES=<archs>` | CUDA GPU architectures to compile for (semicolon-separated, e.g., `75;80;86;90`) | `75;80;86;90` |
| `-DCMAKE_HIP_ARCHITECTURES=<archs>` | HIP GPU architectures to compile for (semicolon-separated) | `gfx942` |
| `-DHIP_COMPILER_FLAGS=<compiler_flags>` | HIP compiler flags (semicolon-separated) | `-Rpass-analysis=kernel-resource-usage;--gpu-max-threads-per-block=256` |
| `-DCMAKE_BUILD_TYPE=Type` | Build type (`Release`, `Debug`, `RelWithDebInfo`, `MinSizeRel`) | (CMake default) |
| `-DENABLE_TESTS=ON/OFF` | Enable FastEddy model tests (includes CTest and the `tests/` subdirectory) | `OFF` |
| `-DENABLE_COVERAGE=ON/OFF` | Enable host-side coverage data generation (requires GNU/Clang compiler; generates LCOV coverage targets when combined with `ENABLE_TESTS`) | `OFF` |

#### Example Configurations

Basic build with defaults:
```bash
mkdir build && cd build
cmake ..
```

Build with GAD extension and specific GPU architectures:
```bash
mkdir build && cd build
cmake .. -DWITH_GAD=ON -DCMAKE_CUDA_ARCHITECTURES="80;86"
```

Build with both extensions enabled in Debug mode:
```bash
mkdir build && cd build
cmake .. -DWITH_GAD=ON -DWITH_URBAN=ON -DCMAKE_BUILD_TYPE=Debug
```

Build with tests and coverage enabled:
```bash
mkdir build && cd build
cmake .. -DENABLE_TESTS=ON -DENABLE_COVERAGE=ON
```

### Building

Once configured, compile the project:

```bash
make -j$(nproc)
```

The executable `FastEddy_model` will be produced in the `build/` directory.

### Cleaning

To clean the build artifacts:

```bash
rm -rf build
```

## Documentation
[FastEddy documentation](https://fasteddy-model.readthedocs.io/) for this version and previous versions are available through Read the Docs.

## Tutorials 
FastEddy tutorials for idealized cases are available in the [Tutorials](https://fasteddy-model.readthedocs.io/en/latest/Tutorials/index.html) section of the documentation.

## Publications
FastEddy publications are available in the [Publications](https://fasteddy-model.readthedocs.io/en/latest/publications.html) section of the documentation.
