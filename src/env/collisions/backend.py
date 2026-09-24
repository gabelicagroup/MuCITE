"""Deterministic bundled/approximate IonSPA backend selection."""

from __future__ import annotations

import warnings
from importlib import import_module
from pathlib import Path
from typing import Any, Optional

from .constants import EXPECTED_BUNDLED_VERSION
from .models import IonSpaLoadError


def _canonical_module_origin(module: Any, canonical_dir: Path) -> Path:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise IonSpaLoadError("Canonical bundled IonSPA has no module origin")
    origin = Path(module_file).resolve()
    try:
        origin.relative_to(canonical_dir.resolve())
    except ValueError as exc:
        raise IonSpaLoadError(
            f"Refusing non-canonical IonSPA origin {origin}; "
            f"expected {canonical_dir.resolve()}"
        ) from exc
    return origin


def _validate_version_and_api(module: Any) -> None:
    get_version = getattr(module, "get_version", None)
    version = get_version() if callable(get_version) else None
    if version != EXPECTED_BUNDLED_VERSION:
        raise IonSpaLoadError(
            "Unsupported bundled IonSPA version "
            f"{version!r}; expected {EXPECTED_BUNDLED_VERSION!r}"
        )
    _validate_public_api(module)


def _validate_public_api(module: Any) -> None:
    missing_api = [
        name
        for name in ("ionclass", "fracloss", "cellclass")
        if not callable(getattr(module, name, None))
    ]
    if missing_api:
        raise IonSpaLoadError(
            "IonSPA provider is missing callable API: "
            + ", ".join(missing_api)
        )


def _validate_resource(canonical_dir: Path) -> None:
    required_resource = canonical_dir / "hcprofiles2.json"
    if not required_resource.is_file():
        raise IonSpaLoadError(
            f"Canonical bundled IonSPA resource is missing: {required_resource}"
        )


class IonSpaBackendMixin:
    """Backend loading, validation, and provenance reporting."""

    def _load_ionspa(self) -> tuple[Optional[Any], Optional[Exception]]:
        if self.backend == "approximate":
            warnings.warn(
                "IonSPA approximate backend explicitly selected; its thermodynamic "
                "and pseudo-atom models are not equivalent to bundled IonSPA.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None, None
        if self.backend == "local":
            return self._load_local_ionspa(), None
        raise IonSpaLoadError(
            "IonSPA is not bundled in the public MuCITE distribution. "
            "Use --collision-physics-backend iict-lite with explicit parameters, "
            "or --collision-physics-backend ionspa --ionspa-backend local "
            "with your separately obtained lawful provider. "
            "Legacy configurations are not silently converted."
        )

    def _load_local_ionspa(self) -> Any:
        try:
            module = import_module("ionspa")
        except (ImportError, ModuleNotFoundError) as exc:
            raise IonSpaLoadError(
                "The optional local IonSPA backend was selected, but no "
                "importable licensed 'ionspa' package is available. Install "
                "or expose your lawful local copy, or select iict-lite."
            ) from exc
        _validate_public_api(module)
        return module

    def _validate_bundled_module(self, module: Any) -> None:
        _canonical_module_origin(module, self.canonical_package_dir)
        _validate_version_and_api(module)
        _validate_resource(self.canonical_package_dir.resolve())

    @property
    def runtime_info(self) -> dict[str, Any]:
        """Return report-safe backend and provenance information."""

        if self.ionspa is None:
            version = None
            origin = None
            implementation = "src.env.collisions.models.ApproximateIonModel"
        else:
            get_version = getattr(self.ionspa, "get_version", None)
            version = get_version() if callable(get_version) else None
            origin = str(Path(self.ionspa.__file__).resolve())
            implementation = self.ionspa.__name__
        canonical_metadata = False  # Public distribution has no bundled provider.
        return {
            "requested_backend": self.backend,
            "effective_backend": self.backend,
            "loaded": self.ionspa is not None,
            "implementation": implementation,
            "version": version,
            "origin": origin,
            "canonical_package_dir": (
                str(self.canonical_package_dir.resolve())
                if canonical_metadata
                else None
            ),
            "approximation_enabled": self.backend == "approximate",
            "provenance_metadata": (
                str((self.canonical_package_dir / "UPSTREAM.json").resolve())
                if canonical_metadata
                else None
            ),
            "source_url": None,
            "source_revision": None,
        }


__all__ = ["IonSpaBackendMixin", "import_module"]
