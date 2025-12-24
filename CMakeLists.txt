cmake_minimum_required(VERSION 3.8)
project(rl)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Finding packages
find_package(Eigen3 3 REQUIRED NO_MODULE)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Define the module
pybind11_add_module(rl_env
    src/bindings.cpp
    src/QLearning.cpp
    src/EpsilonGreedy.cpp
    src/MonteCarloSearch.cpp
)

# Link Eigen to the pybind11 module
target_link_libraries(rl_env PRIVATE Eigen3::Eigen)

# Add include directories if you have custom headers
target_include_directories(rl_env PRIVATE "${CMAKE_SOURCE_DIR}/include")

# Add compile options
target_compile_options(rl_env PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -flto>
    $<$<CONFIG:Debug>:-O0 -g>
)
