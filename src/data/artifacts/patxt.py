"""Strict streaming parser for retained two-dimensional SIMION PATXT files."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union

import numpy as np

from .patxt_models import (
    PatxtHeader,
    PatxtMaskGrid,
    PatxtPotentialGrid,
    _PatxtParseState,
)

_UNSEEN_ELECTRODE_ID = np.iinfo(np.int32).min

def _header_int(raw_header: Mapping[str, str], key: str, path: Path) -> int:
    if key not in raw_header:
        raise ValueError(f"PATXT header is missing required key '{key}': {path}")
    try:
        return int(raw_header[key])
    except ValueError as exc:
        raise ValueError(
            f"PATXT header key '{key}' must be an integer in {path}; "
            f"got {raw_header[key]!r}."
        ) from exc


def _header_float(
    raw_header: Mapping[str, str],
    key: str,
    path: Path,
    *,
    required: bool,
    default: float = 0.0,
) -> float:
    if key not in raw_header:
        if required:
            raise ValueError(f"PATXT header is missing required key '{key}': {path}")
        return float(default)
    try:
        value = float(raw_header[key])
    except ValueError as exc:
        raise ValueError(
            f"PATXT header key '{key}' must be numeric in {path}; "
            f"got {raw_header[key]!r}."
        ) from exc
    if not np.isfinite(value):
        raise ValueError(
            f"PATXT header key '{key}' must be finite in {path}; got {value!r}."
        )
    return value


def _optional_header_int(
    raw_header: Mapping[str, str],
    key: str,
    path: Path,
) -> int:
    return _header_int(raw_header, key, path) if key in raw_header else 0


def _parse_patxt_header(raw_header: dict[str, str], path: Path) -> PatxtHeader:
    nx = _header_int(raw_header, "nx", path)
    ny = _header_int(raw_header, "ny", path)
    nz = _header_int(raw_header, "nz", path)
    if nx <= 1 or ny <= 1 or nz != 1:
        raise ValueError(
            f"Unsupported 2D PATXT grid in {path}: nx={nx}, ny={ny}, nz={nz}; "
            "expected nx>1, ny>1, nz=1."
        )
    grids_per_mm = _header_float(raw_header, "ng", path, required=True)
    if grids_per_mm <= 0.0:
        raise ValueError(
            f"PATXT header ng must be positive in {path}; got {grids_per_mm}."
        )
    return PatxtHeader(
        raw=dict(raw_header),
        mode=_optional_header_int(raw_header, "mode", path),
        symmetry=raw_header.get("symmetry", ""),
        field_type=raw_header.get("field_type", ""),
        data_format=raw_header.get("data_format", ""),
        fast_adjustable=_optional_header_int(raw_header, "fast_adjustable", path),
        mirror_x=_optional_header_int(raw_header, "mirror_x", path),
        mirror_y=_optional_header_int(raw_header, "mirror_y", path),
        mirror_z=_optional_header_int(raw_header, "mirror_z", path),
        nx=nx,
        ny=ny,
        nz=nz,
        grids_per_mm=grids_per_mm,
        max_voltage_v=_header_float(
            raw_header,
            "max_voltage",
            path,
            required=False,
        ),
    )


def _initialize_storage(
    state: _PatxtParseState,
    payload: str,
) -> None:
    assert state.header is not None
    shape = (state.header.ny, state.header.nx)
    if payload == "potential":
        state.potential_v = np.full(shape, np.nan, dtype=float)
        state.electrode_mask = np.zeros(shape, dtype=bool)
        return
    if payload == "mask":
        state.metal_mask = np.zeros(shape, dtype=bool)
        state.electrode_id = np.full(
            shape,
            _UNSEEN_ELECTRODE_ID,
            dtype=np.int32,
        )
        return
    raise RuntimeError(f"Unsupported PATXT payload: {payload}")


def _consume_array_marker(
    line: str,
    line_number: int,
    path: Path,
    state: _PatxtParseState,
) -> bool:
    if line == "begin_potential_array":
        if state.section != "before_header":
            raise ValueError(
                f"Unexpected begin_potential_array at line {line_number} in {path}."
            )
        return True
    if line == "end_potential_array":
        if state.section != "after_points":
            raise ValueError(
                f"Unexpected end_potential_array at line {line_number} in {path}."
            )
        return True
    return False


def _consume_header_marker(
    line: str,
    line_number: int,
    path: Path,
    payload: str,
    state: _PatxtParseState,
) -> bool:
    if line == "begin_header":
        if state.section != "before_header" or state.saw_begin_header:
            raise ValueError(
                f"Duplicate or out-of-order begin_header at line "
                f"{line_number} in {path}."
            )
        state.saw_begin_header = True
        state.section = "header"
        return True
    if line != "end_header":
        return False
    if state.section != "header":
        raise ValueError(
            f"Unexpected end_header at line {line_number} in {path}."
        )
    state.header = _parse_patxt_header(state.raw_header, path)
    state.saw_end_header = True
    state.section = "after_header"
    _initialize_storage(state, payload)
    return True


def _consume_points_marker(
    line: str,
    line_number: int,
    path: Path,
    state: _PatxtParseState,
) -> bool:
    if line == "begin_points":
        if (
            state.section != "after_header"
            or state.header is None
            or state.saw_begin_points
        ):
            raise ValueError(
                f"PATXT entered points before one complete header at "
                f"line {line_number} in {path}."
            )
        state.saw_begin_points = True
        state.section = "points"
        return True
    if line != "end_points":
        return False
    if state.section != "points":
        raise ValueError(
            f"Unexpected end_points at line {line_number} in {path}."
        )
    state.saw_end_points = True
    state.section = "after_points"
    return True


def _consume_header_row(
    line: str,
    line_number: int,
    path: Path,
    state: _PatxtParseState,
) -> None:
    parts = line.split(None, 1)
    if len(parts) != 2:
        raise ValueError(
            f"Malformed PATXT header row at line {line_number} "
            f"in {path}: {line!r}."
        )
    key, value = parts
    if key in state.raw_header:
        raise ValueError(
            f"Duplicate PATXT header key '{key}' at line "
            f"{line_number} in {path}."
        )
    state.raw_header[key] = value


def _parse_point_row(
    line: str,
    line_number: int,
    path: Path,
    header: PatxtHeader,
) -> tuple[int, int, int, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(
            f"Malformed PATXT point row at line {line_number} in "
            f"{path}: expected 5 columns "
            "'x y z is_electrode potential', got "
            f"{len(parts)} ({line!r})."
        )
    try:
        x_i, y_i, z_i = (int(parts[index]) for index in range(3))
        is_electrode_value = int(parts[3])
        point_potential_v = float(parts[4])
    except ValueError as exc:
        raise ValueError(
            f"Malformed numeric PATXT point row at line "
            f"{line_number} in {path}: {line!r}."
        ) from exc
    if not np.isfinite(point_potential_v):
        raise ValueError(
            f"Non-finite PATXT potential at line {line_number} "
            f"in {path}: {point_potential_v!r}."
        )
    if z_i != 0:
        raise ValueError(
            f"PATXT point at line {line_number} in {path} has "
            f"z={z_i}; a 2D nz=1 file requires z=0."
        )
    if not (0 <= x_i < header.nx and 0 <= y_i < header.ny):
        raise ValueError(
            f"PATXT point index out of bounds at line "
            f"{line_number} in {path}: (x={x_i}, y={y_i}), "
            f"grid nx={header.nx}, ny={header.ny}."
        )
    return x_i, y_i, is_electrode_value, point_potential_v


def _store_potential_point(
    state: _PatxtParseState,
    x_i: int,
    y_i: int,
    is_electrode_value: int,
    point_potential_v: float,
    line_number: int,
    path: Path,
) -> None:
    assert state.potential_v is not None
    assert state.electrode_mask is not None
    if not np.isnan(state.potential_v[y_i, x_i]):
        raise ValueError(
            f"Duplicate PATXT point (x={x_i}, y={y_i}, z=0) "
            f"at line {line_number} in {path}."
        )
    state.potential_v[y_i, x_i] = point_potential_v
    state.electrode_mask[y_i, x_i] = is_electrode_value != 0


def _store_mask_point(
    state: _PatxtParseState,
    x_i: int,
    y_i: int,
    is_electrode_value: int,
    point_potential_v: float,
    line_number: int,
    path: Path,
) -> None:
    assert state.electrode_id is not None
    assert state.metal_mask is not None
    if state.electrode_id[y_i, x_i] != _UNSEEN_ELECTRODE_ID:
        raise ValueError(
            f"Duplicate PATXT point (x={x_i}, y={y_i}, z=0) "
            f"at line {line_number} in {path}."
        )
    is_metal = is_electrode_value != 0
    state.metal_mask[y_i, x_i] = is_metal
    if not is_metal:
        state.electrode_id[y_i, x_i] = -1
        return
    rounded_id = int(round(point_potential_v))
    if not np.iinfo(np.int32).min < rounded_id <= np.iinfo(np.int32).max:
        raise ValueError(
            f"PATXT electrode id is outside int32 range at "
            f"line {line_number} in {path}: {rounded_id}."
        )
    state.electrode_id[y_i, x_i] = rounded_id


def _consume_point_row(
    line: str,
    line_number: int,
    path: Path,
    state: _PatxtParseState,
) -> None:
    if state.header is None:  # pragma: no cover - state invariant
        raise RuntimeError("PATXT points encountered without a header.")
    values = _parse_point_row(line, line_number, path, state.header)
    if state.potential_v is not None:
        _store_potential_point(state, *values, line_number, path)
    else:
        _store_mask_point(state, *values, line_number, path)
    state.point_count += 1


def _consume_line(
    line: str,
    line_number: int,
    path: Path,
    payload: str,
    state: _PatxtParseState,
) -> None:
    if _consume_array_marker(line, line_number, path, state):
        return
    if _consume_header_marker(line, line_number, path, payload, state):
        return
    if _consume_points_marker(line, line_number, path, state):
        return
    if state.section == "header":
        _consume_header_row(line, line_number, path, state)
        return
    if state.section == "points":
        _consume_point_row(line, line_number, path, state)
        return
    raise ValueError(
        f"Unexpected PATXT content at line {line_number} in {path}: {line!r}."
    )


def _first_missing_point(state: _PatxtParseState) -> tuple[int, int]:
    if state.potential_v is not None:
        unseen = np.isnan(state.potential_v)
    elif state.electrode_id is not None:
        unseen = state.electrode_id == _UNSEEN_ELECTRODE_ID
    else:  # pragma: no cover - internal invariant
        raise RuntimeError("PATXT storage was not initialized.")
    for y_i in range(unseen.shape[0]):
        missing_x = np.flatnonzero(unseen[y_i])
        if missing_x.size:
            return int(missing_x[0]), int(y_i)
    return -1, -1


def _validate_complete(path: Path, state: _PatxtParseState) -> None:
    if (
        not state.saw_begin_header
        or not state.saw_end_header
        or state.header is None
    ):
        raise ValueError(f"PATXT file is missing a complete header section: {path}")
    if not state.saw_begin_points:
        raise ValueError(f"PATXT file is missing begin_points: {path}")
    expected_count = state.header.nx * state.header.ny
    if not state.saw_end_points:
        raise ValueError(
            f"PATXT points section is truncated in {path}: missing end_points "
            f"after {state.point_count} of {expected_count} points."
        )
    if state.point_count == expected_count:
        return
    missing_x, missing_y = _first_missing_point(state)
    raise ValueError(
        f"PATXT point grid is incomplete in {path}: parsed {state.point_count} "
        f"unique points, expected {expected_count}, missing "
        f"{expected_count - state.point_count}; first missing point is "
        f"(x={missing_x}, y={missing_y}, z=0)."
    )


def _build_result(
    payload: str,
    state: _PatxtParseState,
) -> Union[PatxtPotentialGrid, PatxtMaskGrid]:
    assert state.header is not None
    if payload == "potential":
        assert state.potential_v is not None
        assert state.electrode_mask is not None
        return PatxtPotentialGrid(
            header=state.header,
            potential_v=state.potential_v,
            electrode_mask=state.electrode_mask,
            point_count=state.point_count,
        )
    assert state.metal_mask is not None
    assert state.electrode_id is not None
    return PatxtMaskGrid(
        header=state.header,
        metal_mask=state.metal_mask,
        electrode_id=state.electrode_id,
        point_count=state.point_count,
    )


def _read_patxt_2d(
    path: Path,
    *,
    payload: str,
) -> Union[PatxtPotentialGrid, PatxtMaskGrid]:
    path = Path(path)
    state = _PatxtParseState()
    try:
        handle = path.open("r", encoding="utf-8-sig", errors="strict")
    except OSError as exc:
        raise OSError(f"Cannot open PATXT file {path}: {exc}") from exc
    try:
        with handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                _consume_line(line, line_number, path, payload, state)
    except UnicodeError as exc:
        raise ValueError(
            f"PATXT file is not valid UTF-8/ASCII text: {path}"
        ) from exc
    _validate_complete(path, state)
    return _build_result(payload, state)


def read_patxt_potential_grid(path: Path) -> PatxtPotentialGrid:
    parsed = _read_patxt_2d(path, payload="potential")
    if not isinstance(parsed, PatxtPotentialGrid):  # pragma: no cover
        raise RuntimeError("PATXT parser returned the wrong payload type.")
    return parsed


def read_patxt_mask_grid(path: Path) -> PatxtMaskGrid:
    parsed = _read_patxt_2d(path, payload="mask")
    if not isinstance(parsed, PatxtMaskGrid):  # pragma: no cover
        raise RuntimeError("PATXT parser returned the wrong payload type.")
    return parsed


__all__ = [
    "PatxtHeader",
    "PatxtMaskGrid",
    "PatxtPotentialGrid",
    "read_patxt_mask_grid",
    "read_patxt_potential_grid",
]
