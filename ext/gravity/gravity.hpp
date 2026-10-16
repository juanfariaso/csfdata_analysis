#pragma once

#include <pybind11/numpy.h>

namespace py = pybind11;

using FloatArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

py::array_t<double> distance_matrix(const FloatArray& positions);

py::array_t<double> gravitational_potential(
    const FloatArray& positions,
    const FloatArray& masses,
    double gravitational_constant
);

bool openmp_enabled();

int openmp_max_threads();
