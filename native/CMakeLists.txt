cmake_minimum_required(VERSION 3.16)

project(kb_pente_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KB_PENTE_ENABLE_SANITIZERS "Build native targets with ASan and UBSan" OFF)
option(KB_PENTE_ENABLE_TSAN
    "Build native targets with ThreadSanitizer" OFF)
option(KB_PENTE_ENABLE_BENCHMARKS
    "Build opt-in native CPU scaling benchmarks" OFF)

if(KB_PENTE_ENABLE_TSAN AND KB_PENTE_ENABLE_SANITIZERS)
    message(FATAL_ERROR "KB_PENTE_ENABLE_TSAN cannot be combined with KB_PENTE_ENABLE_SANITIZERS")
endif()

find_package(Threads REQUIRED)

add_library(kb_pente_tsan_options INTERFACE)
if(KB_PENTE_ENABLE_TSAN AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kb_pente_tsan_options INTERFACE
        -fsanitize=thread
        -fno-omit-frame-pointer
        -fno-pie
    )
    target_link_options(kb_pente_tsan_options INTERFACE
        -fsanitize=thread
        -no-pie
    )
endif()

add_library(kb_pente_core STATIC
    src/position.cpp
    src/position_hash.cpp
    src/rules.cpp
    src/game.cpp
)

target_compile_features(kb_pente_core PUBLIC cxx_std_17)
target_link_libraries(kb_pente_core PUBLIC kb_pente_tsan_options)
target_include_directories(kb_pente_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(MSVC)
    target_compile_options(kb_pente_core PRIVATE /W4 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kb_pente_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kb_pente_core PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(kb_pente_core PRIVATE -fsanitize=address,undefined)
endif()

add_library(kb_pente_parallel STATIC
    src/parallel/worker_pool.cpp
)
target_link_libraries(kb_pente_parallel PUBLIC Threads::Threads)
target_link_libraries(kb_pente_parallel PUBLIC kb_pente_tsan_options)
target_compile_features(kb_pente_parallel PUBLIC cxx_std_17)
target_include_directories(kb_pente_parallel
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(MSVC)
    target_compile_options(kb_pente_parallel PRIVATE /W4 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kb_pente_parallel PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kb_pente_parallel PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(kb_pente_parallel PRIVATE
        -fsanitize=address,undefined
    )
endif()

add_library(kb_pente_mcts STATIC
    src/mcts/inference_workspace.cpp
    src/mcts/tree_arena.cpp
    src/mcts/tree.cpp
    src/mcts/search_session.cpp
    src/mcts/search_batch.cpp
)
target_link_libraries(kb_pente_mcts PUBLIC
    kb_pente_core
    kb_pente_parallel
    kb_pente_tsan_options
)
target_compile_features(kb_pente_mcts PUBLIC cxx_std_17)
target_include_directories(kb_pente_mcts
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

if(MSVC)
    target_compile_options(kb_pente_mcts PRIVATE /W4 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kb_pente_mcts PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(KB_PENTE_ENABLE_BENCHMARKS)
    add_executable(kb_pente_native_worker_benchmark
        bench/worker_pool_mcts_benchmark.cpp
    )
    target_link_libraries(kb_pente_native_worker_benchmark
        PRIVATE
            kb_pente_mcts
            kb_pente_parallel
    )
    target_compile_features(kb_pente_native_worker_benchmark PUBLIC cxx_std_17)
    target_include_directories(kb_pente_native_worker_benchmark
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    if(MSVC)
        target_compile_options(kb_pente_native_worker_benchmark PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_worker_benchmark PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_worker_benchmark PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_worker_benchmark PRIVATE
            -fsanitize=address,undefined
        )
    endif()

endif()

if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kb_pente_mcts PRIVATE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(kb_pente_mcts PRIVATE
        -fsanitize=address,undefined
    )
endif()

include(CTest)

if(BUILD_TESTING)
    add_executable(kb_pente_native_tests
        tests/core_test.cpp
    )
    target_link_libraries(kb_pente_native_tests PRIVATE kb_pente_core)

    add_executable(kb_pente_native_position_hash_tests
        tests/position_hash_test.cpp
    )
    target_link_libraries(
        kb_pente_native_position_hash_tests
        PRIVATE kb_pente_mcts
    )

    add_executable(kb_pente_native_diff_runner
        tests/differential_runner.cpp
    )
    target_link_libraries(kb_pente_native_diff_runner PRIVATE kb_pente_core)

    add_executable(kb_pente_native_mcts_diff_runner
        tests/mcts_differential_runner.cpp
    )
    target_link_libraries(kb_pente_native_mcts_diff_runner PRIVATE kb_pente_mcts)

    add_executable(kb_pente_native_mcts_tests
        tests/mcts_test.cpp
    )
    target_link_libraries(kb_pente_native_mcts_tests PRIVATE kb_pente_mcts)

    add_executable(kb_pente_native_search_tests
        tests/search_test.cpp
    )
    target_link_libraries(kb_pente_native_search_tests PRIVATE kb_pente_mcts)

    add_executable(kb_pente_native_search_session_tests
        tests/search_session_test.cpp
    )
    target_link_libraries(kb_pente_native_search_session_tests PRIVATE kb_pente_mcts)

    add_executable(kb_pente_native_advance_tests
        tests/root_advance_test.cpp
    )
    target_link_libraries(kb_pente_native_advance_tests PRIVATE kb_pente_mcts)

    add_executable(kb_pente_native_telemetry_tests
        tests/telemetry_test.cpp
    )
    target_link_libraries(kb_pente_native_telemetry_tests PRIVATE kb_pente_mcts)

    add_executable(kb_pente_native_search_batch_tests
        tests/search_batch_test.cpp
    )
    target_link_libraries(kb_pente_native_search_batch_tests PRIVATE kb_pente_mcts)

    add_executable(kb_pente_native_inference_workspace_tests
        tests/inference_workspace_test.cpp
    )
    target_link_libraries(
        kb_pente_native_inference_workspace_tests
        PRIVATE kb_pente_mcts
    )

    add_executable(kb_pente_native_worker_pool_tests
        tests/worker_pool_test.cpp
    )
    target_link_libraries(kb_pente_native_worker_pool_tests PRIVATE kb_pente_parallel)

    if(MSVC)
        target_compile_options(kb_pente_native_tests PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(MSVC)
        target_compile_options(
            kb_pente_native_position_hash_tests
            PRIVATE /W4 /permissive-
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(
            kb_pente_native_position_hash_tests
            PRIVATE -Wall -Wextra -Wpedantic
        )
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_tests PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_position_hash_tests PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_position_hash_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()

    if(MSVC)
        target_compile_options(kb_pente_native_diff_runner PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_diff_runner PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(MSVC)
        target_compile_options(kb_pente_native_mcts_diff_runner PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_mcts_diff_runner PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_diff_runner PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_diff_runner PRIVATE
            -fsanitize=address,undefined
        )
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_mcts_diff_runner PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_mcts_diff_runner PRIVATE
            -fsanitize=address,undefined
        )
    endif()

    if(MSVC)
        target_compile_options(kb_pente_native_mcts_tests PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_mcts_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(MSVC)
        target_compile_options(kb_pente_native_search_tests PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_search_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(MSVC)
        target_compile_options(kb_pente_native_search_session_tests PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_search_session_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(MSVC)
        target_compile_options(kb_pente_native_advance_tests PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_advance_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(MSVC)
        target_compile_options(kb_pente_native_telemetry_tests PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_telemetry_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(MSVC)
        target_compile_options(kb_pente_native_search_batch_tests PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_search_batch_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(MSVC)
        target_compile_options(
            kb_pente_native_inference_workspace_tests
            PRIVATE /W4 /permissive-
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(
            kb_pente_native_inference_workspace_tests
            PRIVATE -Wall -Wextra -Wpedantic
        )
    endif()

    if(MSVC)
        target_compile_options(kb_pente_native_worker_pool_tests PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_worker_pool_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_mcts_tests PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_mcts_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_search_tests PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_search_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_search_session_tests PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_search_session_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_advance_tests PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_advance_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_telemetry_tests PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_telemetry_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_search_batch_tests PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_search_batch_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_inference_workspace_tests PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_inference_workspace_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()

    if(KB_PENTE_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_worker_pool_tests PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(kb_pente_native_worker_pool_tests PRIVATE
            -fsanitize=address,undefined
        )
    endif()

    add_test(NAME kb_pente_native_core COMMAND kb_pente_native_tests)
    add_test(
        NAME kb_pente_native_position_hash
        COMMAND kb_pente_native_position_hash_tests
    )
    add_test(NAME kb_pente_native_arena COMMAND kb_pente_native_mcts_tests)
    add_test(NAME kb_pente_native_search COMMAND kb_pente_native_search_tests)
    add_test(
        NAME kb_pente_native_search_session
        COMMAND kb_pente_native_search_session_tests
    )
    add_test(NAME kb_pente_native_advance COMMAND kb_pente_native_advance_tests)
    add_test(NAME kb_pente_native_telemetry COMMAND kb_pente_native_telemetry_tests)
    add_test(NAME kb_pente_native_search_batch COMMAND kb_pente_native_search_batch_tests)
    add_test(
        NAME kb_pente_native_inference_workspace
        COMMAND kb_pente_native_inference_workspace_tests
    )
    add_test(NAME kb_pente_native_worker_pool COMMAND kb_pente_native_worker_pool_tests)
    set_tests_properties(kb_pente_native_worker_pool PROPERTIES TIMEOUT 60)

    if(KB_PENTE_ENABLE_BENCHMARKS)
        add_test(
            NAME kb_pente_native_worker_benchmark_smoke
            COMMAND kb_pente_native_worker_benchmark
                --trees 2
                --simulations 2
                --board-size 5
                --repetitions 1
                --warmups 0
                --workers 1,2
        )
        set_tests_properties(
            kb_pente_native_worker_benchmark_smoke
            PROPERTIES TIMEOUT 30
        )
        add_test(
            NAME kb_pente_native_worker_benchmark_invalid_configuration
            COMMAND kb_pente_native_worker_benchmark --trees 0
        )
        set_tests_properties(
            kb_pente_native_worker_benchmark_invalid_configuration
            PROPERTIES
                TIMEOUT 30
                WILL_FAIL TRUE
        )
        add_test(
            NAME kb_pente_native_worker_benchmark_invalid_workers
            COMMAND kb_pente_native_worker_benchmark --workers 2,4
        )
        set_tests_properties(
            kb_pente_native_worker_benchmark_invalid_workers
            PROPERTIES
                TIMEOUT 30
                WILL_FAIL TRUE
        )
        add_test(
            NAME kb_pente_native_worker_benchmark_invalid_board_size
            COMMAND kb_pente_native_worker_benchmark --board-size 4
        )
        set_tests_properties(
            kb_pente_native_worker_benchmark_invalid_board_size
            PROPERTIES
                TIMEOUT 30
                WILL_FAIL TRUE
        )
    endif()

    set(KB_PENTE_DIFFERENTIAL_PYTHON "" CACHE FILEPATH
        "Python interpreter with the project dependencies for differential tests")
    if(NOT KB_PENTE_DIFFERENTIAL_PYTHON)
        if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Scripts/python.exe")
            set(KB_PENTE_DIFFERENTIAL_PYTHON
                "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/Scripts/python.exe")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python")
            set(KB_PENTE_DIFFERENTIAL_PYTHON
                "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python")
        endif()
    endif()
    if(KB_PENTE_DIFFERENTIAL_PYTHON)
        add_test(
            NAME kb_pente_native_differential
            COMMAND "${KB_PENTE_DIFFERENTIAL_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/../script/native_game_differential.py"
                --runner $<TARGET_FILE:kb_pente_native_diff_runner>
        )
        set_tests_properties(kb_pente_native_differential PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
            TIMEOUT 600
        )
        add_test(
            NAME kb_pente_native_mcts_differential
            COMMAND "${KB_PENTE_DIFFERENTIAL_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/../script/native_mcts_differential.py"
                --runner $<TARGET_FILE:kb_pente_native_mcts_diff_runner>
        )
        set_tests_properties(kb_pente_native_mcts_differential PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
            TIMEOUT 600
        )
    else()
        message(STATUS
            "Skipping kb_pente_native_differential: set KB_PENTE_DIFFERENTIAL_PYTHON to a project Python")
    endif()
endif()
