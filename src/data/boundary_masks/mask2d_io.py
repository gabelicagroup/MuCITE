"""File adapters for axisymmetric electrode-mask models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ...env.boundaries.mask2d import ElectrodeMask, finalize_electrode_mask
from ..artifacts import read_patxt_mask_grid
from .metadata import json_ready


DEFAULT_GEM_MASK_PATXT_PATH = Path("E_field") / "slens100x_gem.patxt"
DEFAULT_GEM_MASK_PATXT_PATHS = (
    DEFAULT_GEM_MASK_PATXT_PATH,
    Path("E_field") / "slens" / "slens100x_gem.patxt",
)


def _load_electrode_mask_npz(path: Path) -> ElectrodeMask:
    with np.load(path, allow_pickle=True) as data:
        metadata = (
            json.loads(str(data["metadata_json"].item()))
            if "metadata_json" in data
            else {}
        )
        metadata["source_format"] = metadata.get("source_format", "npz")
        metadata["cache_path"] = str(path)
        return finalize_electrode_mask(
            r_coords_m=np.asarray(data["r_coords_m"], dtype=float),
            z_coords_m=np.asarray(data["z_coords_m"], dtype=float),
            metal_mask=np.asarray(data["metal_mask"], dtype=bool),
            electrode_id=np.asarray(data["electrode_id"], dtype=np.int32),
            surface_distance_m=np.asarray(
                data["surface_distance_m"], dtype=np.float32
            ),
            metadata=metadata,
        )


def _save_electrode_mask_npz(path: Path, mask: ElectrodeMask) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        r_coords_m=mask.r_coords_m,
        z_coords_m=mask.z_coords_m,
        metal_mask=mask.metal_mask.astype(np.uint8),
        electrode_id=mask.electrode_id.astype(np.int32),
        surface_distance_m=mask.surface_distance_m.astype(np.float32),
        metadata_json=json.dumps(
            json_ready(mask.metadata),
            sort_keys=True,
        ),
    )


def _patxt_metadata(
    path: Path,
    parsed: Any,
    z_offset_m: float,
    z_offset_source: str,
) -> dict[str, Any]:
    header = parsed.header
    return {
        "source_path": str(path),
        "source_format": "pa_text_mask",
        "coordinate_mapping": (
            "PA text x index -> global z, y index -> global r"
        ),
        "z_offset_m": float(z_offset_m),
        "z_offset_source": z_offset_source,
        "pa_header_ng": header.grids_per_mm,
        "pa_header_nx": header.nx,
        "pa_header_ny": header.ny,
        "pa_header_nz": header.nz,
        "pa_header_fast_adjustable": header.fast_adjustable,
        "pa_point_count": parsed.point_count,
        "pa_expected_point_count": header.nx * header.ny,
        "pa_integrity_validated": True,
        "electrode_id_source": (
            "rounded electrode potential in PA text electrode points"
        ),
    }


def _load_electrode_mask_patxt(
    path: Path,
    *,
    z_offset_m: float = 0.0,
    z_offset_source: str = "cli",
) -> ElectrodeMask:
    parsed = read_patxt_mask_grid(Path(path))
    header = parsed.header
    pitch_m = 1.0e-3 / header.grids_per_mm
    return finalize_electrode_mask(
        r_coords_m=np.arange(header.ny, dtype=float) * pitch_m,
        z_coords_m=(
            z_offset_m + np.arange(header.nx, dtype=float) * pitch_m
        ),
        metal_mask=parsed.metal_mask,
        electrode_id=parsed.electrode_id,
        surface_distance_m=None,
        metadata=_patxt_metadata(
            path,
            parsed,
            z_offset_m,
            z_offset_source,
        ),
    )


def _infer_mask_z_offset_m(
    field_metadata: Optional[dict[str, Any]],
) -> float:
    if not isinstance(field_metadata, dict):
        return 0.0
    simion_metadata = field_metadata.get("simion", field_metadata)
    if not isinstance(simion_metadata, dict):
        return 0.0
    dc_offset = float(simion_metadata.get("dc_z_offset_m", 0.0))
    rf_offset = float(simion_metadata.get("rf_z_offset_m", dc_offset))
    if abs(dc_offset - rf_offset) > 1.0e-12:
        raise ValueError(
            "Cannot infer electrode mask z offset because baked DC/RF "
            f"offsets differ: dc_z_offset_m={dc_offset}, "
            f"rf_z_offset_m={rf_offset}"
        )
    return dc_offset


def resolve_default_electrode_mask_path(
    project_root: Optional[Path] = None,
) -> Optional[Path]:
    candidates: list[Path] = []
    if project_root is not None:
        candidates.extend(
            Path(project_root) / relative_path
            for relative_path in DEFAULT_GEM_MASK_PATXT_PATHS
        )
    candidates.extend(DEFAULT_GEM_MASK_PATXT_PATHS)
    return next(
        (candidate for candidate in candidates if candidate.exists()),
        None,
    )


def _valid_cached_mask(
    source_path: Path,
    cache_path: Optional[Path],
    requested_z_offset_m: float,
) -> Optional[ElectrodeMask]:
    if cache_path is None:
        return None
    cache_path = Path(cache_path)
    if (
        not cache_path.exists()
        or cache_path.stat().st_mtime < source_path.stat().st_mtime
    ):
        return None
    cache = _load_electrode_mask_npz(cache_path)
    cached_offset_m = float(cache.metadata.get("z_offset_m", 0.0))
    offset_matches = (
        source_path.suffix.lower() != ".patxt"
        or abs(cached_offset_m - requested_z_offset_m) <= 1.0e-12
    )
    if not offset_matches:
        return None
    cache.metadata["source_path"] = str(source_path)
    cache.metadata["loaded_from_cache"] = True
    return cache


def _load_mask_source(
    source_path: Path,
    requested_z_offset_m: float,
    z_offset_source: str,
) -> ElectrodeMask:
    suffix = source_path.suffix.lower()
    if suffix == ".npz":
        return _load_electrode_mask_npz(source_path)
    if suffix == ".patxt":
        return _load_electrode_mask_patxt(
            source_path,
            z_offset_m=requested_z_offset_m,
            z_offset_source=z_offset_source,
        )
    raise ValueError(
        f"Unsupported electrode mask format {source_path.suffix}; "
        "use .patxt or .npz."
    )


def load_electrode_mask(
    *,
    source_path: Optional[Path],
    cache_path: Optional[Path] = None,
    z_offset_m: Optional[float] = None,
    field_metadata: Optional[dict[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> Optional[ElectrodeMask]:
    resolved_source = (
        resolve_default_electrode_mask_path(project_root)
        if source_path is None
        else source_path
    )
    if resolved_source is None:
        return None
    path = Path(resolved_source)
    if not path.exists():
        raise FileNotFoundError(
            f"Electrode mask path does not exist: {path}"
        )
    requested_offset_m = (
        _infer_mask_z_offset_m(field_metadata)
        if z_offset_m is None
        else float(z_offset_m)
    )
    cached = _valid_cached_mask(path, cache_path, requested_offset_m)
    if cached is not None:
        return cached
    offset_source = "baked_field_metadata" if z_offset_m is None else "cli"
    mask = _load_mask_source(path, requested_offset_m, offset_source)
    if cache_path is not None:
        output_path = Path(cache_path)
        _save_electrode_mask_npz(output_path, mask)
        mask.metadata["cache_path"] = str(output_path)
        mask.metadata["loaded_from_cache"] = False
    return mask


__all__ = [
    "DEFAULT_GEM_MASK_PATXT_PATH",
    "DEFAULT_GEM_MASK_PATXT_PATHS",
    "load_electrode_mask",
    "resolve_default_electrode_mask_path",
]
