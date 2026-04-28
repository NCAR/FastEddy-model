#
# Basic help test
add_test(NAME FastEddy_Help COMMAND $<TARGET_FILE:FastEddy_model>)
set_tests_properties(FastEddy_Help PROPERTIES 
    PASS_REGULAR_EXPRESSION "usage:.*paramFile")


#
# Integration tests
# Drive the fasteddy executable for basic integration quantification


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
    set(TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/${test_name}")
    set(TEST_INPUT "${TEST_DIR}/${test_name}.in")

    file(MAKE_DIRECTORY "${TEST_DIR}")
    file(MAKE_DIRECTORY "${TEST_DIR}/test_output")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.in"
        "${TEST_INPUT}"
        COPYONLY
    )

    add_test(
        NAME integration.${test_name}
        COMMAND $<TARGET_FILE:FastEddy_model> "${TEST_INPUT}"
    )
    set_tests_properties(integration.${test_name} PROPERTIES
        WORKING_DIRECTORY "${TEST_DIR}"
        LABELS "integration;gpu"
        TIMEOUT 300
    )
endforeach()
