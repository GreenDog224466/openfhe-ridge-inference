cmake_minimum_required(VERSION 3.16)

project(RidgeInferencePrototype CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ridge_demo src/ridge.cpp)

# M1 Flags
target_compile_options(ridge_demo PRIVATE -O3 -ffast-math -Wall -Wextra)

# --- MANUAL OVERRIDE FIX ---
target_include_directories(ridge_demo PRIVATE 
    /usr/local/include/openfhe
    /usr/local/include/openfhe/core
    /usr/local/include/openfhe/pke
    /usr/local/include/openfhe/binfhe  # <--- ADDED THIS LINE
)

# Link Libraries
target_link_libraries(ridge_demo
    /usr/local/lib/libOPENFHEpke.dylib
    /usr/local/lib/libOPENFHEcore.dylib
    /usr/local/lib/libOPENFHEbinfhe.dylib 
)

message(STATUS "Manual configuration using /usr/local/include/openfhe with binfhe support")