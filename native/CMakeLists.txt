cmake_minimum_required(VERSION 3.16)

project(kb_pente_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KB_PENTE_ENABLE_SANITIZERS "Build native targets with ASan and UBSan" OFF)

add_library(kb_pente_core STATIC
    src/position.cpp
    src/rules.cpp
    src/game.cpp
)

target_compile_features(kb_pente_core PUBLIC cxx_std_17)
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

add_library(kb_pente_mcts STATIC
    src/mcts/tree_arena.cpp
    src/mcts/tree.cpp
    src/mcts/search_session.cpp
)
target_link_libraries(kb_pente_mcts PUBLIC kb_pente_core)
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

    add_executable(kb_pente_native_diff_runner
        tests/differential_runner.cpp
    )
    target_link_libraries(kb_pente_native_diff_runner PRIVATE kb_pente_core)

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

    if(MSVC)
        target_compile_options(kb_pente_native_tests PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_tests PRIVATE -Wall -Wextra -Wpedantic)
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

    if(MSVC)
        target_compile_options(kb_pente_native_diff_runner PRIVATE /W4 /permissive-)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(kb_pente_native_diff_runner PRIVATE -Wall -Wextra -Wpedantic)
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

    add_test(NAME kb_pente_native_core COMMAND kb_pente_native_tests)
    add_test(NAME kb_pente_native_arena COMMAND kb_pente_native_mcts_tests)
    add_test(NAME kb_pente_native_search COMMAND kb_pente_native_search_tests)
    add_test(
        NAME kb_pente_native_search_session
        COMMAND kb_pente_native_search_session_tests
    )
    add_test(NAME kb_pente_native_advance COMMAND kb_pente_native_advance_tests)
    add_test(NAME kb_pente_native_telemetry COMMAND kb_pente_native_telemetry_tests)

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
    else()
        message(STATUS
            "Skipping kb_pente_native_differential: set KB_PENTE_DIFFERENTIAL_PYTHON to a project Python")
    endif()
endif()
