cmake_minimum_required(VERSION 3.16)

option(
    ACE_PX4_XRCE_PUBLISHER_ONLY
    "Build only the native PX4 ArmJointState publisher"
    OFF
)

if(ACE_PX4_XRCE_PUBLISHER_ONLY)
    project(ace_px4_xrce VERSION 0.1.0 LANGUAGES C)

    include(CTest)
    include(GNUInstallDirs)

    find_package(microcdr 2.0.1 EXACT REQUIRED CONFIG)
    find_package(microxrcedds_client 2.4.0 EXACT REQUIRED CONFIG)

    add_library(ace_px4_arm_joint_state STATIC
        src/arm_joint_state.c
    )
    target_include_directories(ace_px4_arm_joint_state PUBLIC include)
    target_compile_features(ace_px4_arm_joint_state PUBLIC c_std_99)
    target_compile_options(
        ace_px4_arm_joint_state PRIVATE -Wall -Wextra -Wpedantic -Werror
    )

    add_executable(ace-px4-xrce-publisher
        src/publisher.c
    )
    target_compile_features(ace-px4-xrce-publisher PRIVATE c_std_99)
    target_compile_options(
        ace-px4-xrce-publisher PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
    target_compile_definitions(ace-px4-xrce-publisher PRIVATE
        ACE_PX4_SCHEMA_SHA256="c5f5fdf23d08235dfd20f5e39907c4703160c1d2a94ff6233234550ed33744b4"
    )
    target_link_libraries(ace-px4-xrce-publisher PRIVATE
        ace_px4_arm_joint_state
        microxrcedds_client
        microcdr
    )

    install(
        TARGETS ace-px4-xrce-publisher
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    configure_file(manifest.json.in manifest.json @ONLY)
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/manifest.json
        DESTINATION ${CMAKE_INSTALL_DATADIR}/ace-px4-xrce
    )

    if(BUILD_TESTING)
        add_executable(
            ace_px4_arm_joint_state_test
            tests/arm_joint_state_test.c
        )
        target_compile_features(ace_px4_arm_joint_state_test PRIVATE c_std_99)
        target_compile_options(
            ace_px4_arm_joint_state_test
            PRIVATE -Wall -Wextra -Wpedantic -Werror
        )
        target_link_libraries(
            ace_px4_arm_joint_state_test PRIVATE ace_px4_arm_joint_state
        )
        add_test(
            NAME arm_joint_state_wire_validation
            COMMAND ace_px4_arm_joint_state_test
        )
    endif()

    return()
endif()

project(ace_robot_zmq_xrce_stack NONE)

include(ExternalProject)

set(
    ACETELE_XRCE_PREFIX
    "$ENV{HOME}/.local/lib/acetele/xrce-2.4.2"
    CACHE PATH
    "Isolated install prefix for the pinned Agent, Client, and publisher"
)
get_filename_component(ACETELE_XRCE_PREFIX "${ACETELE_XRCE_PREFIX}" ABSOLUTE)
get_filename_component(repository_root "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(common_cmake_args
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${ACETELE_XRCE_PREFIX}
    -DCMAKE_PREFIX_PATH=${ACETELE_XRCE_PREFIX}
)

ExternalProject_Add(microcdr
    GIT_REPOSITORY https://github.com/eProsima/Micro-CDR.git
    GIT_TAG v2.0.1
    GIT_SHALLOW TRUE
    CMAKE_ARGS
        ${common_cmake_args}
        -DBUILD_SHARED_LIBS=OFF
        -DUCDR_BUILD_TESTS=OFF
        -DUCDR_ISOLATED_INSTALL=OFF
)

ExternalProject_Add(micro_xrce_client
    SOURCE_DIR ${repository_root}/third_party/micro_xrce_dds_client
    CMAKE_ARGS
        ${common_cmake_args}
        -DUCLIENT_SUPERBUILD=OFF
        -DUCLIENT_BUILD_TESTS=OFF
        -DUCLIENT_BUILD_EXAMPLES=OFF
        -DUCLIENT_PROFILE_UDP=ON
        -DUCLIENT_PROFILE_TCP=OFF
        -DUCLIENT_PROFILE_SERIAL=OFF
        -DUCLIENT_PROFILE_DISCOVERY=OFF
        -DUCLIENT_PROFILE_CUSTOM_TRANSPORT=OFF
        -DBUILD_SHARED_LIBS=OFF
    DEPENDS microcdr
)

ExternalProject_Add(fastcdr
    GIT_REPOSITORY https://github.com/eProsima/Fast-CDR.git
    GIT_TAG v1.1.1
    GIT_SHALLOW TRUE
    CMAKE_ARGS
        ${common_cmake_args}
        -DBUILD_SHARED_LIBS=OFF
        -DBUILD_DOCUMENTATION=OFF
)

ExternalProject_Add(foonathan_memory
    GIT_REPOSITORY https://github.com/foonathan/memory.git
    GIT_TAG v0.7-3
    GIT_SHALLOW TRUE
    CMAKE_ARGS
        ${common_cmake_args}
        -DFOONATHAN_MEMORY_BUILD_EXAMPLES=OFF
        -DFOONATHAN_MEMORY_BUILD_TESTS=OFF
        -DFOONATHAN_MEMORY_BUILD_TOOLS=ON
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
)

ExternalProject_Add(fastdds
    GIT_REPOSITORY https://github.com/eProsima/Fast-DDS.git
    GIT_TAG v2.12.2
    GIT_SHALLOW TRUE
    CMAKE_ARGS
        ${common_cmake_args}
        -DBUILD_SHARED_LIBS=OFF
        -DTHIRDPARTY=ON
        -DCOMPILE_TOOLS=OFF
        -DSECURITY=OFF
        -DSHM_TRANSPORT_DEFAULT=OFF
        -DBUILD_DOCUMENTATION=OFF
    DEPENDS fastcdr foonathan_memory
)

ExternalProject_Add(micro_xrce_agent
    SOURCE_DIR ${repository_root}/third_party/micro_xrce_dds_agent
    CMAKE_ARGS
        ${common_cmake_args}
        -DUAGENT_SUPERBUILD=OFF
        -DUAGENT_BUILD_TESTS=OFF
        -DUAGENT_P2P_PROFILE=OFF
        -DUAGENT_SOCKETCAN_PROFILE=OFF
        -DUAGENT_DISCOVERY_PROFILE=OFF
        -DUAGENT_LOGGER_PROFILE=OFF
        -DUAGENT_USE_SYSTEM_FASTDDS=ON
        -DUAGENT_USE_SYSTEM_FASTCDR=ON
        -DBUILD_SHARED_LIBS=OFF
    DEPENDS fastcdr foonathan_memory fastdds
)

ExternalProject_Add(ace_px4_xrce_publisher
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    CMAKE_ARGS
        ${common_cmake_args}
        -DACE_PX4_XRCE_PUBLISHER_ONLY=ON
        -DBUILD_TESTING=ON
    TEST_COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    TEST_AFTER_INSTALL TRUE
    DEPENDS microcdr micro_xrce_client micro_xrce_agent
)
