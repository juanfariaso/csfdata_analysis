pybind11_add_module(
  gravity
  MODULE
  module.cpp
  gravity.cpp
)

target_compile_features(gravity PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(gravity PRIVATE /O2)
else()
  target_compile_options(gravity PRIVATE -O3)
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(gravity PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
  # Apple Clang does not advertise Homebrew's libomp through the standard
  # compiler probe. Use the two conventional Homebrew prefixes when present.
  find_path(
    LIBOMP_INCLUDE_DIR
    omp.h
    HINTS /opt/homebrew/opt/libomp /usr/local/opt/libomp
    PATH_SUFFIXES include
  )
  find_library(
    LIBOMP_LIBRARY
    omp
    HINTS /opt/homebrew/opt/libomp /usr/local/opt/libomp
    PATH_SUFFIXES lib
  )
  if(LIBOMP_INCLUDE_DIR AND LIBOMP_LIBRARY)
    target_compile_options(gravity PRIVATE -Xpreprocessor -fopenmp)
    target_include_directories(gravity PRIVATE ${LIBOMP_INCLUDE_DIR})
    target_link_libraries(gravity PRIVATE ${LIBOMP_LIBRARY})
  else()
    message(STATUS "libomp not found; gravity extension will run serially")
  endif()
endif()

install(TARGETS gravity LIBRARY DESTINATION csfdata_analysis/extensions)
