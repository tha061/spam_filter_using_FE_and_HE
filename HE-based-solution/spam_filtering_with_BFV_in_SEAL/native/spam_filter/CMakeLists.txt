# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.12)

project(SEALSpamFilter VERSION 3.6.2 LANGUAGES CXX)

# If not called from root CMakeLists.txt
if(NOT DEFINED SEAL_BUILD_SPAMFILTER)
    set(SEAL_BUILD_SPAMFILTER ON)

    # Import Microsoft SEAL
    find_package(SEAL 3.6.2 EXACT REQUIRED)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
endif()

if(SEAL_BUILD_SPAMFILTER)
    add_executable(sealspamfilter)

    target_sources(sealspamfilter
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/examples.cpp
            #${CMAKE_CURRENT_LIST_DIR}/1_bfv_basics.cpp
            #${CMAKE_CURRENT_LIST_DIR}/2_encoders.cpp
            #${CMAKE_CURRENT_LIST_DIR}/3_levels.cpp
            #${CMAKE_CURRENT_LIST_DIR}/4_ckks_basics.cpp
            #${CMAKE_CURRENT_LIST_DIR}/5_rotation.cpp
            #${CMAKE_CURRENT_LIST_DIR}/6_serialization.cpp
            #${CMAKE_CURRENT_LIST_DIR}/7_performance.cpp
            ${CMAKE_CURRENT_LIST_DIR}/total_sum.cpp
            ${CMAKE_CURRENT_LIST_DIR}/mat_mul_bfv.cpp
            ${CMAKE_CURRENT_LIST_DIR}/bfv_spam_filter.cpp
            ${CMAKE_CURRENT_LIST_DIR}/spam_filter_plaintext.cpp
            ${CMAKE_CURRENT_LIST_DIR}/spam_filter_ckks.cpp
            ${CMAKE_CURRENT_LIST_DIR}/trec07_spam_filter.cpp
            ${CMAKE_CURRENT_LIST_DIR}/bfv_encrypted_email_plain_model.cpp
    )

    if(TARGET SEAL::seal)
        target_link_libraries(sealspamfilter PRIVATE SEAL::seal)
    elseif(TARGET SEAL::seal_shared)
        target_link_libraries(sealspamfilter PRIVATE SEAL::seal_shared)
    else()
        message(FATAL_ERROR "Cannot find target SEAL::seal or SEAL::seal_shared")
    endif()
endif()
