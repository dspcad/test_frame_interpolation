cmake_minimum_required (VERSION 3.8 FATAL_ERROR)
project(similarity_check LANGUAGES CXX CUDA)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

include_directories( include )

add_subdirectory (src)
add_subdirectory (demo)

