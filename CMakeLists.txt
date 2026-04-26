cmake_minimum_required(VERSION 3.21)

# ------------------------------------------------------------------
# GPU Backend Selection
# ------------------------------------------------------------------
# Choose between NVIDIA CUDA and AMD HIP/ROCm backends.
#   - CUDA  (default):  Uses NVIDIA CUDA toolkit and nvcc compiler.
#   - HIP   (optional): Uses AMD ROCm toolkit and hipcc compiler.
#
# Usage examples:
#   cmake ..                              # CUDA backend (default)
#   cmake .. -DUSE_HIP=ON                 # HIP/ROCm backend
# ------------------------------------------------------------------
option(USE_HIP "Use AMD HIP/ROCm backend instead of NVIDIA CUDA" OFF)
option(ENABLE_TESTS "Enable FastEddy model tests" OFF)

# ------------------------------------------------------------------
# Project Declaration (language depends on backend)
# ------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)  # it be great to move to 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(USE_HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    project(FastEddy-model LANGUAGES C CXX HIP)
    message(STATUS "GPU Backend: AMD HIP/ROCm")
else()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    project(FastEddy-model LANGUAGES C CXX CUDA)
    message(STATUS "GPU Backend: NVIDIA CUDA")
endif()

if(ENABLE_TESTS)
    include(CTest)
endif()

# Add local cmake/ directory to module search path (for FindNetCDF.cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ------------------------------------------------------------------
# Find Common Dependencies
# ------------------------------------------------------------------
# MPI
find_package(MPI REQUIRED)

# NetCDF (uses cmake/FindNetCDF.cmake)
find_package(NetCDF REQUIRED COMPONENTS C)

# ------------------------------------------------------------------
# GPU Backend Configuration
# ------------------------------------------------------------------
if(USE_HIP)
    # --- HIP/ROCm ---
    # ROCm is expected to be installed and visible (default /opt/rocm).
    # Users can override via HIP_ROOT or CMAKE_PREFIX_PATH.
    find_package(hip REQUIRED)

    # Set HIP GPU architecture (default: gfx942 for AMD MI300)
    # Override with: -DCMAKE_HIP_ARCHITECTURES=gfx90a
    if(NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "HIP architectures")
        message(STATUS "Defaulting CMAKE_HIP_ARCHITECTURES to ${CMAKE_HIP_ARCHITECTURES}")
    endif()

    # HIP-specific compiler flags (from Makefile.hip)
    # -Rpass-analysis=kernel-resource-usage --gpu-max-threads-per-block=256
    set(HIP_COMPILER_FLAGS "-Rpass-analysis=kernel-resource-usage;--gpu-max-threads-per-block=256"
        CACHE STRING "HIP compiler flags")
    set(CMAKE_HIP_FLAGS "${HIP_COMPILER_FLAGS}")

else()
    # --- NVIDIA CUDA ---
    find_package(CUDAToolkit REQUIRED)

    # Set CUDA architectures.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA architectures")
        message(STATUS "Defaulting CMAKE_CUDA_ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------
option(WITH_GAD "Enable the GAD extension module" OFF)
option(WITH_URBAN "Enable the URBAN extension module" OFF)


# ------------------------------------------------------------------
# Source Files
# ------------------------------------------------------------------
set(CORE_SOURCES
    # Modules (Base/Core Modules)
    SRC/FECUDA/fecuda.c
    SRC/FEMAIN/FastEddy.c
    SRC/FEMPI/fempi.c
    SRC/GRID/grid.c
    SRC/GRID/CUDA/cuda_grid.c
    SRC/HYDRO_CORE/hydro_core.c
    SRC/HYDRO_CORE/CUDA/cuda_hydroCore.c
    SRC/IO/io.c
    SRC/IO/ioVarsList.c
    SRC/MEM_UTILS/mem_utils.c
    SRC/PARAMETERS/parameters.c
    SRC/PARAMETERS/hashTable.c
    SRC/TIME_INTEGRATION/time_integration.c
    SRC/TIME_INTEGRATION/CUDA/cuda_timeInt.c
)

# The legacy build compiles a small set of aggregate translation units and
# pulls the remaining device implementations in with #include "... .cu".
set(GPU_ENTRY_SOURCES
    # FECUDA aggregate TU
    SRC/FECUDA/fecuda_Device.cu

    # GRID
    SRC/GRID/CUDA/cuda_gridDevice.cu

    # HYDRO_CORE aggregate TU
    SRC/HYDRO_CORE/CUDA/cuda_hydroCoreDevice.cu

    # TIME_INTEGRATION aggregate TU
    SRC/TIME_INTEGRATION/CUDA/cuda_timeIntDevice.cu
)

# These CUDA sources are included by the aggregate translation units above.
set(GPU_INCLUDED_SOURCES
    # FECUDA
    SRC/FECUDA/fecuda_PlugIns.cu
    SRC/FECUDA/fecuda_Utils.cu

    # HYDRO_CORE
    SRC/HYDRO_CORE/CUDA/cuda_advectionDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_auxScalarsDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_BaseStateDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_BCsDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_buoyancyDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_canopyDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_cellpertDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_coriolisDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_filtersDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_largeScaleForcingsDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_moistureDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_molecularDiffDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_pressureDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_rayleighDampingDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_sgstkeDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_sgsTurbDevice.cu
    SRC/HYDRO_CORE/CUDA/cuda_surfaceLayerDevice.cu

    # TIME_INTEGRATION
    SRC/TIME_INTEGRATION/CUDA/cuda_RKschemes.cu
)

# Conditionally add extension sources
if(WITH_GAD)
    list(APPEND CORE_SOURCES "SRC/EXTENSIONS/GAD/GAD.c")
    list(APPEND GPU_INCLUDED_SOURCES "SRC/EXTENSIONS/GAD/CUDA/cuda_GADDevice.cu")
endif()

if(WITH_URBAN)
    list(APPEND CORE_SOURCES "SRC/EXTENSIONS/URBAN/urban.c")
    list(APPEND GPU_INCLUDED_SOURCES "SRC/EXTENSIONS/URBAN/CUDA/cuda_urbanDevice.cu")
endif()

set(HIPIFY_CUDA_SOURCES
    ${GPU_ENTRY_SOURCES}
    ${GPU_INCLUDED_SOURCES}
)

# ------------------------------------------------------------------
# Core Executable Target
# ------------------------------------------------------------------
add_executable(FastEddy_model
    ${CORE_SOURCES}
    ${GPU_ENTRY_SOURCES}
)

if(WITH_GAD)
   target_compile_definitions(FastEddy_model PRIVATE GAD_EXT)
endif()

if(WITH_URBAN)
   target_compile_definitions(FastEddy_model PRIVATE URBAN_EXT)
endif()

# Deprecated C++ dialect warnings (from Makefile's DEFINES)
target_compile_definitions(FastEddy_model PRIVATE
    CUB_IGNORE_DEPRECATED_CPP_DIALECT
    THRUST_IGNORE_DEPRECATED_CPP_DIALECT
)
# ------------------------------------------------------------------
# Include Directories
# ------------------------------------------------------------------
target_include_directories(FastEddy_model PRIVATE
    SRC/EXTENSIONS
    SRC/FECUDA
    SRC/FEMAIN
    SRC/FEMPI
    SRC/GRID
    SRC/GRID/CUDA
    SRC/HYDRO_CORE
    SRC/HYDRO_CORE/CUDA
    SRC/IO
    SRC/MEM_UTILS
    SRC/PARAMETERS
    SRC/TIME_INTEGRATION
    SRC/TIME_INTEGRATION/CUDA
)

# Conditionally add extension include paths
if(WITH_GAD)
    target_include_directories(FastEddy_model PRIVATE
        SRC/EXTENSIONS/GAD
        SRC/EXTENSIONS/GAD/CUDA
    )
endif()

if(WITH_URBAN)
    target_include_directories(FastEddy_model PRIVATE
        SRC/EXTENSIONS/URBAN
        SRC/EXTENSIONS/URBAN/CUDA
    )
endif()

# ------------------------------------------------------------------
# Linking
# ------------------------------------------------------------------
if(USE_HIP)
    # HIP/ROCm backend: link against rocPRIM, hiprand, amdhip64, hiprtc
    target_link_libraries(FastEddy_model PRIVATE
        roc::rocprims
        roc::hiprand
        MPI::MPI_C
        NetCDF::NetCDF_C
    )
else()
    # NVIDIA CUDA backend
    target_link_libraries(FastEddy_model PRIVATE
        CUDA::cudart   # CUDA runtime library
        CUDA::curand   # CUDA random number generator
        MPI::MPI_C
        NetCDF::NetCDF_C  # NetCDF I/O (includes + libs via FindNetCDF.cmake)
    )
endif()

# ------------------------------------------------------------------
# GPU-Specific Properties
# ------------------------------------------------------------------
if(USE_HIP)
    # HIP: Enable relocatable device code (equivalent to CUDA_SEPARABLE_COMPILATION)
    set_target_properties(FastEddy_model PROPERTIES
        HIP_SEPARABLE_COMPILATION ON
    )
else()
    # CUDA: Enable separable compilation (replaces nvcc -dlink device-link step)
    set_target_properties(FastEddy_model PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ------------------------------------------------------------------
# HIPify Step (if HIP backend selected)
# ------------------------------------------------------------------
if(USE_HIP)
    # hipify-perl converts CUDA source files to HIP-compatible code.
    # The Makefile.hip runs hipify-perl -inplace before compilation.
    # In CMake we create a custom target to hipify sources into the build directory.

    # Find hipify-perl tool
    find_program(HIPIFY_PERL hipify-perl)
    if(NOT HIPIFY_PERL)
        message(FATAL_ERROR "hipify-perl not found. Ensure ROCm is installed and in PATH.")
    endif()

    # Create hipified copies of all CUDA sources used by the aggregate TUs.
    set(HIPIFIED_SOURCES "")
    set(HIPIFIED_GPU_ENTRY_SOURCES "")
    foreach(src_file ${HIPIFY_CUDA_SOURCES})
        get_filename_component(src_dir "${src_file}" DIRECTORY)
        get_filename_component(src_name "${src_file}" NAME)
        set(hipified_file "${CMAKE_CURRENT_BINARY_DIR}/${src_dir}/${src_name}")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${src_dir}")

        add_custom_command(
            OUTPUT "${hipified_file}"
            COMMAND ${HIPIFY_PERL} --default-language=c++ --exclude-backup-file
                    "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}" -o "${hipified_file}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}"
            COMMENT "Hipifying ${src_name}"
            VERBATIM
        )
        list(APPEND HIPIFIED_SOURCES "${hipified_file}")
        list(FIND GPU_ENTRY_SOURCES "${src_file}" src_is_entry)
        if(NOT src_is_entry EQUAL -1)
            list(APPEND HIPIFIED_GPU_ENTRY_SOURCES "${hipified_file}")
        endif()
    endforeach()

    # Also hipify headers that contain CUDA-specific code
    set(HIPIFY_HEADERS
        "SRC/FECUDA/fecuda_Device_cu.h"
        "SRC/FECUDA/fecuda_PlugIns_cu.h"
        "SRC/FECUDA/fecuda_Utils_cu.h"
        "SRC/GRID/CUDA/cuda_gridDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_advectionDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_auxScalarsDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_BaseStateDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_BCsDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_buoyancyDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_canopyDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_cellpertDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_coriolisDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_filtersDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_hydroCoreDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_largeScaleForcingsDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_moistureDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_molecularDiffDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_pressureDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_rayleighDampingDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_sgstkeDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_sgsTurbDevice_cu.h"
        "SRC/HYDRO_CORE/CUDA/cuda_surfaceLayerDevice_cu.h"
        "SRC/TIME_INTEGRATION/CUDA/cuda_RKschemes_cu.h"
        "SRC/TIME_INTEGRATION/CUDA/cuda_timeIntDevice_cu.h"
    )

    foreach(hdr_file ${HIPIFY_HEADERS})
        get_filename_component(src_dir "${hdr_file}" DIRECTORY)
        get_filename_component(src_name "${hdr_file}" NAME)
        set(hipified_hdr "${CMAKE_CURRENT_BINARY_DIR}/${src_dir}/${src_name}")
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${src_dir}")

        add_custom_command(
            OUTPUT "${hipified_hdr}"
            COMMAND ${HIPIFY_PERL} --default-language=c++ --exclude-backup-file
                    "${CMAKE_CURRENT_SOURCE_DIR}/${hdr_file}" -o "${hipified_hdr}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${hdr_file}"
            COMMENT "Hipifying header ${src_name}"
            VERBATIM
        )
        list(APPEND HIPIFIED_SOURCES "${hipified_hdr}")
    endforeach()

    # Replace GPU sources with hipified versions
    # We remove the original .cu files and add the hipified ones
    set_target_properties(FastEddy_model PROPERTIES
        HIP_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Add hipify target as dependency
    add_custom_target(hipify_sources DEPENDS ${HIPIFIED_SOURCES})
    add_dependencies(FastEddy_model hipify_sources)

    # Replace GPU sources: remove originals, add hipified
    # Note: CMake doesn't allow removing sources from a target easily,
    # so we use a custom command approach.
    # The hipified files will be compiled as HIP sources.
    set_source_files_properties(${HIPIFIED_SOURCES} PROPERTIES LANGUAGE HIP)

    # We need to swap the sources. Create a new list without original GPU sources.
    # Since add_executable already happened, we use set_source_files_properties
    # and add the hipified files via a custom approach.
    # Compile the hipified aggregate translation units; they will include the
    # hipified leaf sources from the build tree.
    target_sources(FastEddy_model PRIVATE ${HIPIFIED_GPU_ENTRY_SOURCES})

    # Remove original aggregate CUDA sources from the target
    # This is tricky in CMake. Instead, we mark original .cu files as
    # HEADER_FILE_ONLY so they won't be compiled.
    set_source_files_properties(${GPU_ENTRY_SOURCES} PROPERTIES HEADER_FILE_ONLY TRUE)

    message(STATUS "HIP backend: hipify-perl will convert ${HIPIFY_CUDA_SOURCES} at build time")
endif()

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

message(STATUS "FastEddy model CMake configuration complete.")
