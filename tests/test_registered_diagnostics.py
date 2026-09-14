"""Tests for the generic readiness check for standardized diagnostics."""

from csfdata_analysis.checks import test_diagnostics as run_diagnostic_checks


def test_every_registered_diagnostic_passes_the_readiness_check() -> None:
    """All currently registered diagnostics pass the same built-in command check."""
    checks = run_diagnostic_checks()

    assert checks
    assert all(check.passed for check in checks)
