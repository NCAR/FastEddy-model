find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---------------------------------------------------------------------------
# Python virtual environment
#
# Created once in the build tree.  --system-site-packages lets the venv
# inherit packages already provided by HPC module stacks (e.g. a loaded
# conda/spack environment that already has netCDF4), so pip is only invoked
# when a package is genuinely absent.
# ---------------------------------------------------------------------------
set(_test_venv "${CMAKE_CURRENT_BINARY_DIR}/venv")

if(NOT EXISTS "${_test_venv}/pyvenv.cfg")
    message(STATUS "FastEddy tests: creating Python venv at ${_test_venv}")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m venv --system-site-packages "${_test_venv}"
        RESULT_VARIABLE _venv_rc
        ERROR_VARIABLE  _venv_err
    )
    if(NOT _venv_rc EQUAL 0)
        message(WARNING
            "FastEddy tests: could not create Python venv -- "
            "physics checks will be disabled.\n${_venv_err}")
    endif()
endif()

if(WIN32)
    set(_venv_python "${_test_venv}/Scripts/python.exe")
else()
    set(_venv_python "${_test_venv}/bin/python")
endif()

if(EXISTS "${_venv_python}")
    # Check whether required packages are already importable (covers the
    # --system-site-packages path and avoids a slow pip invocation on every
    # reconfigure).
    execute_process(
        COMMAND "${_venv_python}" -c "import netCDF4, numpy"
        RESULT_VARIABLE _import_rc
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _import_rc EQUAL 0)
        message(STATUS "FastEddy tests: installing netCDF4 + numpy into venv...")
        execute_process(
            COMMAND "${_venv_python}" -m pip install --quiet netCDF4 numpy
            RESULT_VARIABLE _pip_rc
            ERROR_VARIABLE  _pip_err
        )
        if(NOT _pip_rc EQUAL 0)
            message(WARNING
                "FastEddy tests: pip install failed -- "
                "physics checks will be disabled.\n${_pip_err}")
            unset(_venv_python)
        endif()
    endif()
endif()

# Expose as a cache variable so the path survives incremental reconfigures
set(FASTEDDY_TEST_PYTHON "${_venv_python}"
    CACHE INTERNAL "Python interpreter used by FastEddy test scripts")


# ---------------------------------------------------------------------------
# BOMEX initial conditions
#
# tests/initial/FE_BOMEX.0.zip ships with the repo (~280 MB uncompressed).
# Extracted once at configure time into the build tree so that
# integration.test_bomex_mini.50 satisfies its REQUIRED_FILES check.
# ---------------------------------------------------------------------------
set(_bomex_zip "${CMAKE_CURRENT_SOURCE_DIR}/initial/FE_BOMEX.0.zip")
set(_bomex_nc  "${CMAKE_CURRENT_BINARY_DIR}/test_bomex_mini.50/initial/FE_BOMEX.0")

if(EXISTS "${_bomex_zip}" AND NOT EXISTS "${_bomex_nc}")
    message(STATUS "FastEddy tests: extracting BOMEX initial conditions...")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test_bomex_mini.50/initial")
    file(ARCHIVE_EXTRACT
        INPUT       "${_bomex_zip}"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/test_bomex_mini.50/initial"
    )
endif()


# ---------------------------------------------------------------------------
# Basic help test
# ---------------------------------------------------------------------------
add_test(NAME FastEddy_Help COMMAND $<TARGET_FILE:FastEddy_model>)
set_tests_properties(FastEddy_Help PROPERTIES
    PASS_REGULAR_EXPRESSION "usage:.*paramFile")


# ---------------------------------------------------------------------------
# Integration tests + Tier-3 physics conservation checks
#
# For each entry in test_names two tests are registered:
#
#   integration.<name>  -- runs FastEddy; label "integration;gpu"
#   physics.<name>      -- runs check_physics.py on the NetCDF output;
#                          label "physics"
#
# physics.<name> is:
#   • skipped when ioOutputMode=1 (binary output produces no NetCDF)
#   • skipped when the venv setup above failed (FASTEDDY_TEST_PYTHON unset)
#   • ordered after integration.<name> via the DEPENDS property so CTest
#     will not run the check before the model finishes, and will skip it
#     automatically if the model test fails
# ---------------------------------------------------------------------------

# test_<focus>_<size>.<num_steps>
set(test_names
    test_nbl_mini.50
    test_cbl_mini.50
    test_bomex_mini.50
    test_nbl_mini_forcingwrite.50
    test_nbl_mini_binary.50
    test_nbl_mini_auxsc.50
)

foreach(test_name ${test_names})
    set(TEST_DIR   "${CMAKE_CURRENT_BINARY_DIR}/${test_name}")
    set(TEST_INPUT "${TEST_DIR}/${test_name}.in")

    file(MAKE_DIRECTORY "${TEST_DIR}")
    file(MAKE_DIRECTORY "${TEST_DIR}/test_output")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.in"
        "${TEST_INPUT}"
        COPYONLY
    )

    # Detect whether the test needs an external initial-conditions file.
    # Tests that do (e.g. BOMEX) declare  inFile = <name>  in the .in file;
    # without that file present the model will read uninitialised memory and
    # produce all-NaN output.  We use REQUIRED_FILES so CTest skips the test
    # (status "Not Run") rather than running it and recording a false failure.
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.in" _infile_line
         REGEX "^inFile[ \t]*=[ \t]*[^ \t\r\n#]")
    set(_required_file "")
    if(_infile_line)
        string(REGEX REPLACE "^inFile[ \t]*=[ \t]*([^ \t\r\n#]+).*$" "\\1"
               _infile "${_infile_line}")
        file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.in" _inpath_line
             REGEX "^inPath[ \t]*=[ \t]*[^ \t\r\n#]")
        if(_inpath_line)
            string(REGEX REPLACE "^inPath[ \t]*=[ \t]*([^ \t\r\n#]+).*$" "\\1"
                   _inpath "${_inpath_line}")
            # Normalize:  ./foo/  ->  foo
            string(REGEX REPLACE "^\\." "" _inpath "${_inpath}")
            string(REGEX REPLACE "^/" "" _inpath "${_inpath}")
            string(REGEX REPLACE "/$" "" _inpath "${_inpath}")
        else()
            set(_inpath "")
        endif()
        if(_inpath)
            set(_required_file "${TEST_DIR}/${_inpath}/${_infile}")
        else()
            set(_required_file "${TEST_DIR}/${_infile}")
        endif()
    endif()

    add_test(
        NAME integration.${test_name}
        COMMAND $<TARGET_FILE:FastEddy_model> "${TEST_INPUT}"
    )
    set_tests_properties(integration.${test_name} PROPERTIES
        WORKING_DIRECTORY "${TEST_DIR}"
        LABELS            "integration;gpu"
        TIMEOUT           300
    )
    if(_required_file)
        set_tests_properties(integration.${test_name} PROPERTIES
            REQUIRED_FILES "${_required_file}"
        )
    endif()

    # --- physics check ---
    if(FASTEDDY_TEST_PYTHON)
        # Skip tests that write binary output (no NetCDF to check)
        file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.in" _mode_line
             REGEX "^ioOutputMode[ \t]*=")
        if(NOT _mode_line MATCHES "=[ \t]*1")
            # Derive the expected output file name from the .in file
            file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.in" _base_line
                 REGEX "^outFileBase[ \t]*=")
            string(REGEX REPLACE
                "^outFileBase[ \t]*=[ \t]*([^ \t\r\n#]+).*$" "\\1"
                _out_base "${_base_line}")

            add_test(
                NAME physics.${test_name}
                COMMAND "${FASTEDDY_TEST_PYTHON}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/check_physics.py"
                    "${TEST_DIR}/test_output/${_out_base}.0"
            )
            set_tests_properties(physics.${test_name} PROPERTIES
                DEPENDS        integration.${test_name}
                LABELS         "physics"
                TIMEOUT        60
            )
            if(_required_file)
                set_tests_properties(physics.${test_name} PROPERTIES
                    REQUIRED_FILES "${_required_file}"
                )
            endif()
        endif()
    endif()
endforeach()
