"""Standardized diagnostic definitions and automatic module registration."""

from collections.abc import Mapping
from hashlib import sha256
from importlib import import_module, util
from pkgutil import iter_modules
from pathlib import Path

import yaml

from csfdata.catalogue.diagnostics import (
    ChoiceDefinition,
    CollectionDiagnostics,
    collection_diagnostics_path,
    read_collection_diagnostics,
    write_collection_diagnostics,
)
from csfdata_analysis.choices import CHOICES
from csfdata_analysis.diagnostics.base import (
    EvaluationChoice,
    ScalarDiagnostic,
    TimeSeriesDiagnostic,
)


def diagnostic_directories(catalogue_root: Path | str) -> tuple[Path, ...]:
    """Read trusted local diagnostic directories from ``analysis.yaml``.

    Args:
        catalogue_root: Full or lite catalogue root containing an optional
            local ``analysis.yaml`` file.

    Returns:
        Existing local diagnostic directories in their configured order. An
        absent ``analysis.yaml`` produces an empty tuple.

    Raises:
        ValueError: If the configuration schema or directory list is invalid.
        OSError: If the configuration cannot be read.
        yaml.YAMLError: If the configuration is not valid YAML.

    Notes:
        This configuration is local machine setup, not portable catalogue
        metadata. Relative paths are resolved relative to ``analysis.yaml``.
    """
    path = Path(catalogue_root) / "analysis.yaml"
    if not path.is_file():
        return ()
    with path.open(encoding="utf-8") as stream:
        contents = yaml.safe_load(stream)
    if not isinstance(contents, dict) or contents.get("schema_version") != 1:
        raise ValueError("analysis.yaml must use schema_version 1.")
    configured = contents.get("diagnostic_directories")
    if not isinstance(configured, list) or not all(
        isinstance(directory, str) and directory for directory in configured
    ):
        raise ValueError("analysis.yaml must define diagnostic_directories as a list of paths.")

    directories = []
    for configured_directory in configured:
        directory = Path(configured_directory).expanduser()
        if not directory.is_absolute():
            directory = path.parent / directory
        directory = directory.resolve()
        if not directory.is_dir():
            raise ValueError(f"Configured diagnostic directory does not exist: {directory}")
        directories.append(directory)
    return tuple(directories)


def load_diagnostics(
    directories: tuple[Path | str, ...] = (),
) -> dict[
    tuple[str, str], TimeSeriesDiagnostic | ScalarDiagnostic
]:
    """Load and validate built-in and trusted local diagnostic declarations.

    Every module below ``diagnostics/time_series/`` must define a non-empty
    ``DIAGNOSTICS`` tuple containing only :class:`TimeSeriesDiagnostic`
    instances. Every module below ``diagnostics/scalar/`` follows the same
    convention with :class:`ScalarDiagnostic` instances. Every ``.py`` file in
    a configured local directory follows the same tuple convention and may
    contain either diagnostic type.

    Args:
        directories: Trusted local directories whose Python files are loaded in
            addition to built-in diagnostic modules.
    Returns:
        Every declared diagnostic keyed by its stable ``(name, version)``
        identity.

    Raises:
        ValueError: If a configured directory is invalid, a module omits
            ``DIAGNOSTICS``, uses a value other than a non-empty tuple,
            declares a diagnostic of the wrong type, or duplicates a
            name/version identity.
    """
    diagnostics: dict[tuple[str, str], TimeSeriesDiagnostic | ScalarDiagnostic] = {}
    modules = []
    for package_name, expected_type in (
        ("csfdata_analysis.diagnostics.time_series", TimeSeriesDiagnostic),
        ("csfdata_analysis.diagnostics.scalar", ScalarDiagnostic),
    ):
        package = import_module(package_name)
        for module_info in iter_modules(package.__path__, f"{package_name}."):
            if module_info.ispkg:
                raise ValueError(
                    f"Diagnostic package contains an unsupported nested package: "
                    f"{module_info.name}."
                )
            modules.append((module_info.name, import_module(module_info.name), expected_type))
    for configured_directory in directories:
        directory = Path(configured_directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Configured diagnostic directory does not exist: {directory}")
        for path in sorted(directory.glob("*.py")):
            if path.name == "__init__.py":
                continue
            module_name = "csfdata_analysis.local_diagnostics_" + sha256(
                str(path).encode()
            ).hexdigest()
            specification = util.spec_from_file_location(module_name, path)
            if specification is None or specification.loader is None:
                raise ValueError(f"Could not load diagnostic module: {path}")
            module = util.module_from_spec(specification)
            try:
                specification.loader.exec_module(module)
            except Exception as error:
                raise ValueError(
                    f"Could not import diagnostic module {path}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            modules.append((str(path), module, (TimeSeriesDiagnostic, ScalarDiagnostic)))

    for module_name, module, expected_type in modules:
        module_diagnostics = getattr(module, "DIAGNOSTICS", None)
        if not isinstance(module_diagnostics, tuple) or not module_diagnostics:
            raise ValueError(
                f"Diagnostic module {module_name} must define a non-empty DIAGNOSTICS tuple."
            )
        for diagnostic in module_diagnostics:
            if not isinstance(diagnostic, expected_type):
                if isinstance(expected_type, tuple):
                    expected_name = "TimeSeriesDiagnostic or ScalarDiagnostic"
                else:
                    expected_name = expected_type.__name__
                raise ValueError(
                    f"Diagnostic module {module_name} DIAGNOSTICS contains "
                    f"{type(diagnostic).__name__}; expected {expected_name}."
                )
            key = (diagnostic.name, f"v{diagnostic.version}")
            if key in diagnostics:
                raise ValueError(
                    f"Diagnostic module {module_name} duplicates diagnostic "
                    f"{diagnostic.name} {key[1]}."
                )
            diagnostics[key] = diagnostic
    return diagnostics


def time_series_diagnostic(
    name: str,
    directories: tuple[Path | str, ...] = (),
) -> TimeSeriesDiagnostic:
    """Return the latest registered time-series diagnostic with one name.

    Args:
        name: Stable diagnostic name.
        directories: Trusted local directories added to built-in diagnostics.

    Returns:
        The highest available version of the named time-series diagnostic.

    Raises:
        ValueError: If no time-series diagnostic has the requested name.
    """
    matches = [
        diagnostic
        for diagnostic in load_diagnostics(directories).values()
        if isinstance(diagnostic, TimeSeriesDiagnostic) and diagnostic.name == name
    ]
    if not matches:
        raise ValueError(f"Unknown time-series diagnostic: {name}")
    return max(matches, key=lambda diagnostic: diagnostic.version)


def scalar_diagnostic(
    name: str,
    directories: tuple[Path | str, ...] = (),
) -> ScalarDiagnostic:
    """Return the latest registered scalar diagnostic with one name.

    Args:
        name: Stable diagnostic name.
        directories: Trusted local directories added to built-in diagnostics.

    Returns:
        The highest available version of the named scalar diagnostic.

    Raises:
        ValueError: If no scalar diagnostic has the requested name.
    """
    matches = [
        diagnostic
        for diagnostic in load_diagnostics(directories).values()
        if isinstance(diagnostic, ScalarDiagnostic) and diagnostic.name == name
    ]
    if not matches:
        raise ValueError(f"Unknown scalar diagnostic: {name}")
    return max(matches, key=lambda diagnostic: diagnostic.version)


DIAGNOSTICS = load_diagnostics()
"""All diagnostics automatically loaded from their declaring modules."""

TIME_SERIES_DIAGNOSTICS: dict[str, TimeSeriesDiagnostic] = {}
"""Latest version of each time-series diagnostic, selected by name."""

SCALAR_DIAGNOSTICS: dict[str, ScalarDiagnostic] = {}
"""Latest version of each scalar diagnostic, selected by name."""

for registered_diagnostic in DIAGNOSTICS.values():
    selected = (
        TIME_SERIES_DIAGNOSTICS
        if isinstance(registered_diagnostic, TimeSeriesDiagnostic)
        else SCALAR_DIAGNOSTICS
    )
    existing = selected.get(registered_diagnostic.name)
    if existing is None or registered_diagnostic.version > existing.version:
        selected[registered_diagnostic.name] = registered_diagnostic


def ensure_collection_diagnostics(
    collection_root: Path,
    diagnostics: tuple[TimeSeriesDiagnostic | ScalarDiagnostic, ...],
    available_diagnostics: Mapping[
        tuple[str, str], TimeSeriesDiagnostic | ScalarDiagnostic
    ] | None = None,
) -> Path:
    """Ensure a collection contains required analysis definitions and choices.

    Args:
        collection_root: Destination collection directory below ``collections/``.
        diagnostics: Diagnostics requested by a generic analysis runner.
        available_diagnostics: Built-in and trusted local diagnostics available
            to resolve declared requirements. ``None`` uses built-in modules.

    Returns:
        The collection ``diagnostics.yaml`` path, updated only when a required
        definition or choice was absent.

    Raises:
        ValueError: If a diagnostic requests an unregistered choice or
            requirement, or if an existing definition or choice conflicts with
            the registered scientific contract.
        OSError: If the published definitions cannot be read or written.

    Notes:
        This is called automatically by analysis runners. It never replaces an
        existing definition because a name/version conflict could change the
        scientific meaning of already stored data.
    """
    path = collection_diagnostics_path(collection_root)
    existing = read_collection_diagnostics(path) if path.exists() else CollectionDiagnostics((), ())
    registry = DIAGNOSTICS if available_diagnostics is None else available_diagnostics

    # Resolve registered requirements first so the collection declaration stays
    # valid when a scalar diagnostic depends on a time-series product.
    required_diagnostics = []
    pending = list(diagnostics)
    seen_keys: set[tuple[str, str]] = set()
    while pending:
        diagnostic = pending.pop()
        key = (diagnostic.name, f"v{diagnostic.version}")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        required_diagnostics.append(diagnostic)
        for requirement in diagnostic.requires:
            required = registry.get((requirement.name, requirement.version))
            if required is None:
                raise ValueError(
                    "Analysis diagnostic requires an unregistered definition: "
                    f"{requirement.name} {requirement.version}."
                )
            pending.append(required)

    # Preserve unrelated declarations while refusing to silently alter an
    # existing name/version's scientific meaning.
    definitions_by_key = {
        (definition.name, definition.version): definition
        for definition in existing.diagnostics
    }
    changed = False
    for diagnostic in required_diagnostics:
        definition = diagnostic.catalogue_definition()
        key = (definition.name, definition.version)
        existing_definition = definitions_by_key.get(key)
        if existing_definition is None:
            definitions_by_key[key] = definition
            changed = True
        elif existing_definition != definition:
            raise ValueError(
                "Collection diagnostic definition conflicts with registered "
                f"analysis diagnostic: {definition.name} {definition.version}."
            )

    # Choices are shared global scientific conventions, so an existing choice
    # must agree exactly before a new diagnostic may rely on it.
    choices_by_name = {choice.name: choice for choice in existing.choices}
    for diagnostic in required_diagnostics:
        for choice_name in diagnostic.choice_names:
            choice = CHOICES.get(choice_name)
            if choice is None:
                raise ValueError(f"Diagnostic uses an unregistered choice: {choice_name}.")
            expected_choice = ChoiceDefinition(
                choice.name,
                choice.description,
                choice.values,
                choice.default,
            )
            existing_choice = choices_by_name.get(choice_name)
            if existing_choice is None:
                choices_by_name[choice_name] = expected_choice
                changed = True
            elif existing_choice != expected_choice:
                raise ValueError(
                    "Collection choice conflicts with registered analysis choice: "
                    f"{choice_name}."
                )

    # Core validation confirms every retained and newly added cross-reference.
    collection_diagnostics = CollectionDiagnostics(
        tuple(choices_by_name.values()),
        tuple(definitions_by_key.values()),
    )
    if changed:
        write_collection_diagnostics(collection_diagnostics, path)
    return path


__all__ = [
    "EvaluationChoice",
    "DIAGNOSTICS",
    "SCALAR_DIAGNOSTICS",
    "ScalarDiagnostic",
    "TIME_SERIES_DIAGNOSTICS",
    "TimeSeriesDiagnostic",
    "diagnostic_directories",
    "ensure_collection_diagnostics",
    "load_diagnostics",
    "scalar_diagnostic",
    "time_series_diagnostic",
]
