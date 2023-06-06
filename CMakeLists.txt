cmake_minimum_required(VERSION 3.4...3.18)

project(jaxkdtree LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(extern/pybind11)
include_directories("extern/cudaKDTree")

pybind11_add_module(_jaxkdtree src/jaxkdtree.cu)
