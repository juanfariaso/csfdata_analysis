"""Tests for registered analysis choices."""

from csfdata_analysis.choices import CENTER, CHOICES, LAGRANGIAN_RADIUS


def test_registered_choices_have_expected_defaults_and_values() -> None:
    """Keep the initial public choice registry stable and self-consistent."""
    assert CHOICES == {
        "center": CENTER,
        "lagrangian_radius": LAGRANGIAN_RADIUS,
    }
    assert CENTER.default == "stellar_com"
    assert CENTER.values == ("origin", "stellar_com")
    assert LAGRANGIAN_RADIUS.default == "r_l50"
    assert "r_l01" in LAGRANGIAN_RADIUS.values
    assert "r_l95" in LAGRANGIAN_RADIUS.values
