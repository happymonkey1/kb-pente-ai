cmake_minimum_required(VERSION 3.16)

project(kb_pente_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KB_PENTE_ENABLE_SANITIZERS "Build native targets with ASan and UBSan" OFF)

add_library(kb_pente_core STATIC
    src/position.cpp
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

include(CTest)

if(BUILD_TESTING)
    add_executable(kb_pente_native_tests
        tests/core_test.cpp
    )
    target_link_libraries(kb_pente_native_tests PRIVATE kb_pente_core)

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

    add_test(NAME kb_pente_native_core COMMAND kb_pente_native_tests)
endif()
