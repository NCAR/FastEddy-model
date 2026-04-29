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

## Testing

Enable the test suite at configure time with `-DENABLE_TESTS=ON`, then run
tests via CTest from the build directory:

```bash
cmake .. -DENABLE_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

### Test labels

Tests are grouped by label so subsets can be run independently:

| Label | Description |
|---|---|
| `integration` | Runs `FastEddy_model` on a small parameter file; requires a GPU |
| `gpu` | Applied alongside `integration` (same tests) |
| `physics` | Runs `check_physics.py` on the NetCDF output of the preceding integration test |

```bash
# GPU integration tests only
ctest -L integration --output-on-failure

# Physics checks only (assumes integration tests have already produced output)
ctest -L physics --output-on-failure
```

### Physics conservation checks

After each integration test produces a NetCDF output file,
`tests/check_physics.py` validates the following invariants derived from
first principles — no golden-reference data required:

| Check | Invariant |
|---|---|
| Finite fields | No `NaN` or `Inf` in any variable |
| Density positivity | `rho > 0` everywhere |
| Potential temperature | Total `theta` (perturbation + base state) `> 0 K` everywhere |
| TKE non-negativity | Sub-grid TKE `>= -1e-4 m²/s²` (small floor for solver round-off) |
| Moisture bounds | `qv`, `ql >= -1e-6 kg/kg` |
| Velocity bounds | `|u|`, `|v|`, `|w| < 150 m/s` |

The script can also be run directly against any FastEddy NetCDF output:

```bash
python tests/check_physics.py path/to/output.nc
python tests/check_physics.py --verbose path/to/output.nc   # per-field statistics
```

Thresholds can be adjusted via CLI flags; run `python tests/check_physics.py --help`
for the full list.

### Python environment

The physics checks require `netCDF4` and `numpy`.  CMake creates an isolated
Python virtual environment in the build tree at configure time and pip-installs
these packages if they are not already available.  The `--system-site-packages`
flag is set so that packages provided by HPC module stacks (e.g. a loaded
conda or spack environment) are inherited and pip is only invoked for genuinely
absent packages.

If venv creation or pip install fails CMake emits a warning and the `physics.*`
tests are skipped rather than failing.  No manual setup is required.

### BOMEX initial conditions

The `test_bomex_mini.50` test requires an external NetCDF initial conditions
file (`FE_BOMEX.0`, ~280 MB uncompressed).  The compressed file
`tests/initial/FE_BOMEX.0.zip` is included in the repository.  CMake extracts
it into the build tree automatically at configure time; no manual steps are
needed.  If the zip is absent both the integration and physics tests for BOMEX
are reported as *Not Run* rather than failing.

## Documentation
[FastEddy documentation](https://fasteddy-model.readthedocs.io/) for this version and previous versions are available through Read the Docs.

## Tutorials 
FastEddy tutorials for idealized cases are available in the [Tutorials](https://fasteddy-model.readthedocs.io/en/latest/Tutorials/index.html) section of the documentation.

## Publications
FastEddy publications are available in the [Publications](https://fasteddy-model.readthedocs.io/en/latest/publications.html) section of the documentation.
