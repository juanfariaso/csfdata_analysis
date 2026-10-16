cmake_minimum_required(VERSION 3.18...4.1)

project(csfdata_analysis LANGUAGES CXX)

find_package(
  Python
  REQUIRED COMPONENTS Interpreter Development.Module
)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(ext/gravity)
