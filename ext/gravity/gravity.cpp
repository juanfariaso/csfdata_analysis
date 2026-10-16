#include "gravity.hpp"

#include <atomic>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

void validate_positions(const FloatArray& positions) {
    if (positions.ndim() != 2 || positions.shape(1) != 3) {
        throw std::invalid_argument("positions must have shape (n_particles, 3)");
    }

    const auto values = positions.unchecked<2>();
    for (py::ssize_t i = 0; i < values.shape(0); ++i) {
        for (py::ssize_t coordinate = 0; coordinate < 3; ++coordinate) {
            if (!std::isfinite(values(i, coordinate))) {
                throw std::invalid_argument("positions must contain only finite values");
            }
        }
    }
}

}  // namespace

py::array_t<double> distance_matrix(const FloatArray& positions) {
    validate_positions(positions);
    const py::ssize_t count = positions.shape(0);
    py::array_t<double> result({count, count});
    const double* position = positions.data();
    double* distance = result.mutable_data();

    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (py::ssize_t i = 0; i < count; ++i) {
            distance[i * count + i] = 0.0;
            for (py::ssize_t j = i + 1; j < count; ++j) {
                const double dx = position[3 * i] - position[3 * j];
                const double dy = position[3 * i + 1] - position[3 * j + 1];
                const double dz = position[3 * i + 2] - position[3 * j + 2];
                const double separation = std::sqrt(dx * dx + dy * dy + dz * dz);
                distance[i * count + j] = separation;
                distance[j * count + i] = separation;
            }
        }
    }
    return result;
}

py::array_t<double> gravitational_potential(
    const FloatArray& positions,
    const FloatArray& masses,
    const double gravitational_constant
) {
    validate_positions(positions);
    const py::ssize_t count = positions.shape(0);
    if (masses.ndim() != 1 || masses.shape(0) != count) {
        throw std::invalid_argument("masses must have shape (n_particles,)");
    }
    if (!std::isfinite(gravitational_constant) || gravitational_constant < 0.0) {
        throw std::invalid_argument("gravitational_constant must be finite and non-negative");
    }

    const double* mass = masses.data();
    for (py::ssize_t i = 0; i < count; ++i) {
        if (!std::isfinite(mass[i]) || mass[i] < 0.0) {
            throw std::invalid_argument("masses must contain finite non-negative values");
        }
    }

    py::array_t<double> result(count);
    const double* position = positions.data();
    double* potential = result.mutable_data();
    std::atomic<bool> duplicate_position{false};

    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (py::ssize_t i = 0; i < count; ++i) {
            double weighted_inverse_distance = 0.0;
            for (py::ssize_t j = 0; j < count; ++j) {
                if (i == j || mass[j] == 0.0) {
                    continue;
                }
                const double dx = position[3 * i] - position[3 * j];
                const double dy = position[3 * i + 1] - position[3 * j + 1];
                const double dz = position[3 * i + 2] - position[3 * j + 2];
                const double squared_distance = dx * dx + dy * dy + dz * dz;
                if (squared_distance == 0.0) {
                    duplicate_position.store(true, std::memory_order_relaxed);
                    continue;
                }
                weighted_inverse_distance += mass[j] / std::sqrt(squared_distance);
            }
            potential[i] = -gravitational_constant * weighted_inverse_distance;
        }
    }

    if (duplicate_position.load(std::memory_order_relaxed)) {
        throw std::invalid_argument(
            "gravitational potential is undefined for distinct particles at "
            "identical positions"
        );
    }
    return result;
}

bool openmp_enabled() {
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

int openmp_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
