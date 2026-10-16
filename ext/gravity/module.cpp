#include "gravity.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(gravity, module) {
    module.doc() = "Exact compiled gravitational kernels.";

    module.def(
        "distance_matrix",
        &distance_matrix,
        py::arg("positions"),
        "Return the Euclidean pairwise-distance matrix for 3D positions."
    );
    module.def(
        "gravitational_potential",
        &gravitational_potential,
        py::arg("positions"),
        py::arg("masses"),
        py::arg("gravitational_constant") = 1.0,
        "Return each particle's exact point-mass gravitational potential."
    );
    module.def(
        "openmp_enabled",
        &openmp_enabled,
        "Return whether this extension was compiled with OpenMP support."
    );
    module.def(
        "openmp_max_threads",
        &openmp_max_threads,
        "Return the maximum OpenMP threads available to one extension call."
    );
}
