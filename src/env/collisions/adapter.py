"""Public deterministic bridge to bundled or approximate IonSPA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ...config import PROJECT_ROOT
from .backend import IonSpaBackendMixin
from .batch import BatchCollisionMixin
from .cache import IonModelCacheMixin
from .constants import SUPPORTED_IONSPA_BACKENDS
from .single import SingleCollisionMixin
from .thermo import IonSpaThermoMixin


class IonSpaCollisionAdapter(
    BatchCollisionMixin,
    SingleCollisionMixin,
    IonSpaThermoMixin,
    IonModelCacheMixin,
    IonSpaBackendMixin,
):
    """Bridge external ion states to one explicitly selected IonSPA backend."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        *,
        backend: str = "bundled",
    ) -> None:
        self.project_root = Path(project_root or PROJECT_ROOT)
        self.source_dir = Path(__file__).resolve().parent
        self.canonical_package_dir = self.source_dir / "ionspa"
        self.backend = str(backend).strip().lower()
        if self.backend not in SUPPORTED_IONSPA_BACKENDS:
            choices = ", ".join(sorted(SUPPORTED_IONSPA_BACKENDS))
            raise ValueError(
                f"Unsupported IonSPA backend {backend!r}; "
                f"expected one of: {choices}"
            )
        self.ionspa, self.import_error = self._load_ionspa()
        self._ion_model_cache: dict[tuple[Any, ...], Any] = {}
        self._cell_cache: dict[str, Any] = {}


__all__ = ["IonSpaCollisionAdapter"]
