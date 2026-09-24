"""Strict validation for baked axisymmetric grid metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class BakedGridMetadata:
    nr: int
    nz: int
    r_min_m: float
    r_max_m: float
    z_min_m: float
    z_max_m: float
    dr_m: float
    dz_m: float
    r_coords_m: np.ndarray
    z_coords_m: np.ndarray


def _required_grid_value(
    metadata: Mapping[str, Any],
    key: str,
    source: str,
) -> Any:
    if key not in metadata:
        raise ValueError(
            f"Baked grid metadata is missing required key '{key}' in {source}."
        )
    return metadata[key]


def _coerce_coordinate_axis(
    *,
    axis_name: str,
    coords_m: np.ndarray,
    node_count: int,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    coords_m = np.asarray(coords_m, dtype=float)
    if coords_m.ndim != 1:
        raise ValueError(
            f"Baked grid {axis_name}_coords_m must be a 1D array in {source}; "
            f"got shape {coords_m.shape}."
        )
    if len(coords_m) != node_count:
        raise ValueError(
            f"Baked grid {axis_name}_coords_m length mismatch in {source}: "
            f"metadata has n{axis_name}={node_count}, array has "
            f"{len(coords_m)} entries."
        )
    if not np.all(np.isfinite(coords_m)):
        raise ValueError(
            f"Baked grid {axis_name}_coords_m contains non-finite values in "
            f"{source}."
        )
    differences_m = np.diff(coords_m)
    if np.any(differences_m <= 0.0):
        raise ValueError(
            f"Baked grid {axis_name}_coords_m must be strictly increasing in "
            f"{source}."
        )
    return coords_m, differences_m


def _validate_axis_spacing(
    *,
    axis_name: str,
    differences_m: np.ndarray,
    spacing_m: float,
    source: str,
) -> tuple[float, float]:
    coordinate_spacing_m = float(differences_m[0])
    tolerance_m = max(
        1.0e-15,
        abs(spacing_m) * 1.0e-9,
        abs(coordinate_spacing_m) * 1.0e-9,
    )
    if not np.allclose(
        differences_m,
        coordinate_spacing_m,
        rtol=1.0e-9,
        atol=tolerance_m,
    ):
        raise ValueError(
            f"Baked grid {axis_name}_coords_m is not uniformly spaced in "
            f"{source}."
        )
    if not np.isclose(
        coordinate_spacing_m,
        spacing_m,
        rtol=1.0e-9,
        atol=tolerance_m,
    ):
        raise ValueError(
            f"Baked grid {axis_name} coordinate spacing disagrees with "
            f"d{axis_name}_m metadata in {source}: coordinates imply "
            f"{coordinate_spacing_m}, metadata says {spacing_m}."
        )
    return coordinate_spacing_m, tolerance_m


def _validate_axis_endpoints(
    *,
    axis_name: str,
    coords_m: np.ndarray,
    minimum_m: float,
    maximum_m: float,
    spacing_m: float,
    source: str,
) -> float:
    tolerance_m = max(
        1.0e-15,
        abs(maximum_m - minimum_m) * 1.0e-12,
        abs(spacing_m) * 1.0e-9,
    )
    if not np.isclose(coords_m[0], minimum_m, rtol=0.0, atol=tolerance_m):
        raise ValueError(
            f"Baked grid {axis_name}_coords_m start disagrees with "
            f"{axis_name}_min_m metadata in {source}: {coords_m[0]} vs "
            f"{minimum_m}."
        )
    if not np.isclose(coords_m[-1], maximum_m, rtol=0.0, atol=tolerance_m):
        raise ValueError(
            f"Baked grid {axis_name}_coords_m endpoint disagrees with "
            f"{axis_name}_max_m metadata in {source}: {coords_m[-1]} vs "
            f"{maximum_m}."
        )
    return tolerance_m


def _validate_axis_extent(
    *,
    axis_name: str,
    node_count: int,
    minimum_m: float,
    maximum_m: float,
    spacing_m: float,
    tolerance_m: float,
    source: str,
) -> None:
    metadata_extent_m = maximum_m - minimum_m
    implied_extent_m = spacing_m * (node_count - 1)
    if not np.isclose(
        implied_extent_m,
        metadata_extent_m,
        rtol=1.0e-9,
        atol=tolerance_m,
    ):
        raise ValueError(
            f"Baked grid {axis_name} extent is inconsistent in {source}: "
            f"d{axis_name}_m*(n{axis_name}-1)={implied_extent_m}, but "
            f"{axis_name}_max_m-{axis_name}_min_m={metadata_extent_m}."
        )


def _validate_expected_coordinates(
    *,
    axis_name: str,
    coords_m: np.ndarray,
    node_count: int,
    minimum_m: float,
    spacing_m: float,
    tolerance_m: float,
    source: str,
) -> None:
    expected_coords_m = minimum_m + spacing_m * np.arange(
        node_count,
        dtype=float,
    )
    if not np.allclose(
        coords_m,
        expected_coords_m,
        rtol=1.0e-9,
        atol=tolerance_m,
    ):
        raise ValueError(
            f"Baked grid {axis_name}_coords_m disagrees with coordinate "
            f"metadata in {source}."
        )


def _validate_coordinate_axis(
    *,
    axis_name: str,
    coords_m: np.ndarray,
    node_count: int,
    minimum_m: float,
    maximum_m: float,
    spacing_m: float,
    source: str,
) -> np.ndarray:
    coords_m, differences_m = _coerce_coordinate_axis(
        axis_name=axis_name,
        coords_m=coords_m,
        node_count=node_count,
        source=source,
    )
    _, spacing_tolerance_m = _validate_axis_spacing(
        axis_name=axis_name,
        differences_m=differences_m,
        spacing_m=spacing_m,
        source=source,
    )
    endpoint_tolerance_m = _validate_axis_endpoints(
        axis_name=axis_name,
        coords_m=coords_m,
        minimum_m=minimum_m,
        maximum_m=maximum_m,
        spacing_m=spacing_m,
        source=source,
    )
    _validate_axis_extent(
        axis_name=axis_name,
        node_count=node_count,
        minimum_m=minimum_m,
        maximum_m=maximum_m,
        spacing_m=spacing_m,
        tolerance_m=endpoint_tolerance_m,
        source=source,
    )
    _validate_expected_coordinates(
        axis_name=axis_name,
        coords_m=coords_m,
        node_count=node_count,
        minimum_m=minimum_m,
        spacing_m=spacing_m,
        tolerance_m=spacing_tolerance_m,
        source=source,
    )
    return np.ascontiguousarray(coords_m)


def _parse_grid_scalars(
    metadata: Mapping[str, Any],
    source: str,
) -> tuple[int, int, float, float, float, float, float, float]:
    try:
        return (
            int(_required_grid_value(metadata, "nr", source)),
            int(_required_grid_value(metadata, "nz", source)),
            float(_required_grid_value(metadata, "r_min_m", source)),
            float(_required_grid_value(metadata, "r_max_m", source)),
            float(_required_grid_value(metadata, "z_min_m", source)),
            float(_required_grid_value(metadata, "z_max_m", source)),
            float(_required_grid_value(metadata, "dr_m", source)),
            float(_required_grid_value(metadata, "dz_m", source)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Baked grid scalar metadata is invalid in {source}: {exc}"
        ) from exc


def _validate_grid_scalars(
    values: tuple[int, int, float, float, float, float, float, float],
    source: str,
    minimum_nodes: int,
) -> None:
    nr, nz, r_min_m, r_max_m, z_min_m, z_max_m, dr_m, dz_m = values
    if nr < minimum_nodes or nz < minimum_nodes:
        raise ValueError(
            f"Baked grid must contain at least {minimum_nodes} nodes per axis "
            f"in {source}; got nr={nr}, nz={nz}."
        )
    scalar_values = (r_min_m, r_max_m, z_min_m, z_max_m, dr_m, dz_m)
    if not all(np.isfinite(value) for value in scalar_values):
        raise ValueError(
            f"Baked grid scalar metadata contains non-finite values in {source}."
        )
    if r_max_m <= r_min_m or z_max_m <= z_min_m:
        raise ValueError(
            f"Baked grid extents must be positive in {source}: "
            f"r=[{r_min_m}, {r_max_m}], z=[{z_min_m}, {z_max_m}]."
        )
    if dr_m <= 0.0 or dz_m <= 0.0:
        raise ValueError(
            f"Baked grid spacing must be positive in {source}: "
            f"dr_m={dr_m}, dz_m={dz_m}."
        )


def validate_baked_grid_metadata(
    metadata: Mapping[str, Any],
    *,
    source: str = "baked field",
    minimum_nodes: int = 2,
) -> BakedGridMetadata:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Baked grid metadata must be a mapping in {source}.")
    values = _parse_grid_scalars(metadata, source)
    _validate_grid_scalars(values, source, minimum_nodes)
    nr, nz, r_min_m, r_max_m, z_min_m, z_max_m, dr_m, dz_m = values
    r_coords_m = _validate_coordinate_axis(
        axis_name="r",
        coords_m=_required_grid_value(metadata, "r_coords_m", source),
        node_count=nr,
        minimum_m=r_min_m,
        maximum_m=r_max_m,
        spacing_m=dr_m,
        source=source,
    )
    z_coords_m = _validate_coordinate_axis(
        axis_name="z",
        coords_m=_required_grid_value(metadata, "z_coords_m", source),
        node_count=nz,
        minimum_m=z_min_m,
        maximum_m=z_max_m,
        spacing_m=dz_m,
        source=source,
    )
    return BakedGridMetadata(
        nr=nr,
        nz=nz,
        r_min_m=r_min_m,
        r_max_m=r_max_m,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
        dr_m=dr_m,
        dz_m=dz_m,
        r_coords_m=r_coords_m,
        z_coords_m=z_coords_m,
    )


__all__ = ["BakedGridMetadata", "validate_baked_grid_metadata"]
