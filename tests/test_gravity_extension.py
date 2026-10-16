"""Tests for the compiled exact gravitational kernels."""

import os
import subprocess
import sys

import numpy as np
import pytest

from csfdata_analysis.extensions.gravity import (
    distance_matrix,
    gravitational_potential,
    openmp_enabled,
    openmp_max_threads,
)


def test_distance_matrix_returns_euclidean_pairwise_distances() -> None:
    """The public distance kernel returns a symmetric matrix with zero diagonal."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 12.0],
        ]
    )

    result = distance_matrix(positions)

    np.testing.assert_allclose(
        result,
        np.array(
            [
                [0.0, 5.0, 12.0],
                [5.0, 0.0, 13.0],
                [12.0, 13.0, 0.0],
            ]
        ),
    )


def test_gravitational_potential_matches_numpy_direct_sum() -> None:
    """The compiled kernel agrees with an independent exact reference."""
    positions = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [3.0, 0.0, 1.0],
            [0.0, 0.0, -4.0],
        ]
    )
    masses = np.array([1.0, 2.0, 3.0, 4.0])
    gravitational_constant = 0.25
    displacement = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    np.fill_diagonal(distance, np.inf)
    expected = -gravitational_constant * ((1.0 / distance) @ masses)

    result = gravitational_potential(
        positions,
        masses,
        gravitational_constant,
    )

    assert result == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_gravitational_potential_excludes_self_interaction() -> None:
    """A single particle has zero potential from the supplied source set."""
    result = gravitational_potential(
        np.array([[1.0, 2.0, 3.0]]),
        np.array([5.0]),
    )

    assert result.tolist() == [0.0]


def test_openmp_thread_count_uses_standard_environment_variable() -> None:
    """OMP_NUM_THREADS configures each process when OpenMP is available."""
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "2"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from csfdata_analysis.extensions.gravity import "
                "openmp_enabled, openmp_max_threads; "
                "print(int(openmp_enabled()), openmp_max_threads())"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    enabled, maximum = (int(value) for value in completed.stdout.split())

    assert bool(enabled) is openmp_enabled()
    assert maximum == (2 if enabled else 1)
    assert openmp_max_threads() >= 1


@pytest.mark.parametrize(
    ("positions", "masses", "message"),
    [
        (np.zeros((2, 2)), np.ones(2), "positions must have shape"),
        (np.zeros((2, 3)), np.ones(3), "masses must have shape"),
        (
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.ones(2),
            "identical positions",
        ),
    ],
)
def test_gravitational_potential_rejects_invalid_inputs(
    positions: np.ndarray,
    masses: np.ndarray,
    message: str,
) -> None:
    """Invalid geometry and incompatible arrays fail explicitly."""
    with pytest.raises(ValueError, match=message):
        gravitational_potential(positions, masses)
