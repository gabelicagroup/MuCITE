"""离线场数据烘焙器：将 SIMION / Fluent 导出场统一映射到全局 2D 轴对称网格。

核心目标
--------
1. 读取 SIMION 规整网格 CSV 或 Refine 后导出的 ASCII PA 文本（电势场）
2. 读取 Fluent 无序点云 CSV（流场，单位通常已经是 SI）
3. 应用统一的 z 轴对齐偏移量
4. 将所有场插值到统一的全局 `(r, z)` 矩形网格
5. 导出 `baked_fields.npy`，供主程序快速读取
6. 生成诊断图，检查毛细管出口是否被正确对齐

设计约束
--------
- 内部长度单位一律使用米（m）
- 全局坐标系采用 2D 轴对称 `(r, z)`
- 毛细管出口位置写入 baked grid metadata；默认 `z = 5 mm`，可由 CLI 改写
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from scipy.interpolate import RectBivariateSpline, griddata

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ..artifacts.schema import read_patxt_potential_grid

# Ensure CLI help/report text can be printed on Windows consoles even when the
# default code page is not UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


RF_REFERENCE_PEAK_VOLTAGE_V = 1.0
RF_REFERENCE_ODD_ELECTRODE_V = 1.0
RF_REFERENCE_EVEN_ELECTRODE_V = -1.0
RF_REFERENCE_NORMALIZATION_TOLERANCE_V = 1.0e-6
RF_REFERENCE_NORMALIZATION = "adjacent_rf_electrodes_plus_1_v_minus_1_v"
RF_RUNTIME_VOLTAGE_PARAMETER = "rf_peak_voltage_v"
RF_SCALING_EQUATION = (
    "E_rf(t)=E_rf_ref*(rf_peak_voltage_v/rf_reference_peak_voltage_v)"
    "*cos(2*pi*rf_frequency_hz*t+rf_phase_rad)"
)


def _normalize_header(header: str) -> str:
    """把 CSV 表头标准化为仅含字母数字的小写字符串。

    例如：
    - ``z(mm)`` -> ``zmm``
    - ``V_z(m/s)`` -> ``vzms``
    - ``Pressure (Pa)`` -> ``pressurepa``
    """

    return re.sub(r"[^a-z0-9]+", "", header.strip().lower())


def _read_csv_columns(csv_path: Path) -> Dict[str, np.ndarray]:
    """读取 CSV 为列字典。

    这里不用 pandas，保持脚本依赖尽量轻量。
    """

    csv_path = Path(csv_path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        raw_fieldnames = next(reader, None)
        if raw_fieldnames is None:
            raise ValueError(f"CSV 文件缺少表头: {csv_path}")

        normalized_names: list[str] = []
        seen: dict[str, int] = {}
        for raw_name in raw_fieldnames:
            base_name = _normalize_header(raw_name)
            count = seen.get(base_name, 0) + 1
            seen[base_name] = count
            normalized_names.append(base_name if count == 1 else f"{base_name}{count}")

        buckets: Dict[str, list[float]] = {key: [] for key in normalized_names}

        for row in reader:
            for index, normalized_name in enumerate(normalized_names):
                raw_value = row[index] if index < len(row) else ""
                if raw_value is None or raw_value.strip() == "":
                    buckets[normalized_name].append(np.nan)
                else:
                    buckets[normalized_name].append(float(raw_value))

    return {key: np.asarray(values, dtype=float) for key, values in buckets.items()}


def _pick_column(columns: Dict[str, np.ndarray], aliases: Sequence[str], *, csv_path: Path) -> np.ndarray:
    """按别名列表查找列。

    这样可以适应不同软件导出的表头写法。
    """

    for alias in aliases:
        if alias in columns:
            return columns[alias]

    available = ", ".join(sorted(columns.keys()))
    alias_text = ", ".join(aliases)
    if any(name.startswith("daxialvelocity") or name.startswith("dradialvelocity") for name in columns):
        if any("velocity" in alias for alias in aliases):
            raise KeyError(
                f"在 {csv_path} 中没有找到真实速度列 [{alias_text}]。"
                "当前文件包含 daxial-velocity-dx/dradial-velocity-dx 这类速度梯度列，"
                "它们不能替代 V_z/V_r。请从 Fluent 重新导出 Axial Velocity 和 Radial Velocity "
                "（或 X Velocity 和 Y Velocity）分量。可用列为: "
                f"{available}"
            )
    raise KeyError(f"在 {csv_path} 中未找到列 [{alias_text}]；可用列为: {available}")


def _read_simion_patxt(pa_path: Path) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """读取 SIMION ASCII PA 文本文件。

    当前项目中的 `slens_rf.patxt` / `slens_dc.patxt` 不是传统 CSV，而是：

    - 带 `begin_header / end_header` 的头信息
    - 带 `begin_points / end_points` 的规则点阵
    - 每个点为 `x y z is_electrode potential`
    - `potential` 已由 SIMION Refine 得到，可直接作为全空间电势读取

    对当前 ion-guide 文件来说：

    - `symmetry = cylindrical`
    - `nz = 1`
    - 有效二维平面是 `(x, y)`
    - 后续会把 `x -> 全局 z`，`y -> 全局 r`

    返回：

    - `header`: 已做基础类型转换的头信息字典
    - `pa_potential_v`: 形状 `(ny, nx)`，第 0 维是 `y`，第 1 维是 `x`
    - `electrode_mask`: 同形状布尔数组，表示该点是否是电极边界
    """

    parsed = read_patxt_potential_grid(Path(pa_path))
    parsed_header = parsed.header
    header: Dict[str, Any] = {
        "mode": parsed_header.mode,
        "symmetry": parsed_header.symmetry,
        "field_type": parsed_header.field_type,
        "data_format": parsed_header.data_format,
        "fast_adjustable": parsed_header.fast_adjustable,
        "mirror_x": parsed_header.mirror_x,
        "mirror_y": parsed_header.mirror_y,
        "mirror_z": parsed_header.mirror_z,
        "nx": parsed_header.nx,
        "ny": parsed_header.ny,
        "nz": parsed_header.nz,
        "ng_from_header": parsed_header.grids_per_mm,
        "max_voltage": parsed_header.max_voltage_v,
        "point_count": parsed.point_count,
        "integrity_validated": True,
    }
    return header, parsed.potential_v, parsed.electrode_mask


def _rf_reference_metadata() -> Dict[str, Any]:
    """返回 RF 参考场约定的机器可读元数据。"""

    return {
        "rf_normalization": RF_REFERENCE_NORMALIZATION,
        "rf_reference_peak_voltage_v": RF_REFERENCE_PEAK_VOLTAGE_V,
        "rf_reference_odd_electrode_v": RF_REFERENCE_ODD_ELECTRODE_V,
        "rf_reference_even_electrode_v": RF_REFERENCE_EVEN_ELECTRODE_V,
        "rf_reference_adjacent_electrode_delta_v": RF_REFERENCE_ODD_ELECTRODE_V - RF_REFERENCE_EVEN_ELECTRODE_V,
        "rf_runtime_voltage_parameter": RF_RUNTIME_VOLTAGE_PARAMETER,
        "rf_scaling_equation": RF_SCALING_EQUATION,
    }


def _validate_rf_patxt_normalization(
    pa_path: Path,
    pa_potential_v: np.ndarray,
    electrode_mask: np.ndarray,
) -> Dict[str, Any]:
    """确认 RF `.patxt` 是相邻 RF 电极 `+1 V / -1 V` 的归一化基底场。"""

    if not np.any(electrode_mask):
        raise ValueError(
            f"{pa_path} 没有 electrode 标记，无法确认 RF 归一化。"
            "请从 SIMION 导出相邻 RF 电极为 +1 V 和 -1 V 的 refined PA text。"
        )

    electrode_potentials_v = np.asarray(pa_potential_v[electrode_mask], dtype=float)
    finite_electrode_potentials_v = electrode_potentials_v[np.isfinite(electrode_potentials_v)]
    if finite_electrode_potentials_v.size == 0:
        raise ValueError(f"{pa_path} 的电极点电势不是有限数，无法确认 RF 归一化。")

    electrode_min_v = float(np.min(finite_electrode_potentials_v))
    electrode_max_v = float(np.max(finite_electrode_potentials_v))
    finite_potential_v = np.asarray(pa_potential_v[np.isfinite(pa_potential_v)], dtype=float)
    global_min_v = float(np.min(finite_potential_v))
    global_max_v = float(np.max(finite_potential_v))
    has_expected_positive_phase = np.isclose(
        electrode_max_v,
        RF_REFERENCE_ODD_ELECTRODE_V,
        atol=RF_REFERENCE_NORMALIZATION_TOLERANCE_V,
        rtol=0.0,
    )
    has_expected_negative_phase = np.isclose(
        electrode_min_v,
        RF_REFERENCE_EVEN_ELECTRODE_V,
        atol=RF_REFERENCE_NORMALIZATION_TOLERANCE_V,
        rtol=0.0,
    )
    has_expected_global_positive_phase = np.isclose(
        global_max_v,
        RF_REFERENCE_ODD_ELECTRODE_V,
        atol=RF_REFERENCE_NORMALIZATION_TOLERANCE_V,
        rtol=0.0,
    )
    has_expected_global_negative_phase = np.isclose(
        global_min_v,
        RF_REFERENCE_EVEN_ELECTRODE_V,
        atol=RF_REFERENCE_NORMALIZATION_TOLERANCE_V,
        rtol=0.0,
    )
    used_global_fallback = (
        not (has_expected_positive_phase and has_expected_negative_phase)
        and has_expected_global_positive_phase
        and has_expected_global_negative_phase
    )

    if not ((has_expected_positive_phase and has_expected_negative_phase) or used_global_fallback):
        raise ValueError(
            f"{pa_path} 不是当前约定的 RF 归一化基底场："
            f"期望电极电势范围包含 +1 V 和 -1 V，实际 min={electrode_min_v:.12g} V, "
            f"max={electrode_max_v:.12g} V。请在 SIMION 中用相邻 RF 电极 +1 V / -1 V "
            "重新 Refine 并导出 `.patxt`。运行 RF 幅值请在主仿真中用 Vpeak 输入。"
        )

    return {
        "rf_pa_electrode_min_v": electrode_min_v,
        "rf_pa_electrode_max_v": electrode_max_v,
        "rf_pa_global_min_v": global_min_v,
        "rf_pa_global_max_v": global_max_v,
        "rf_pa_normalization_check": "global_potential_range" if used_global_fallback else "electrode_potential_range",
        "rf_pa_normalization_verified": True,
        "rf_pa_normalization_tolerance_v": RF_REFERENCE_NORMALIZATION_TOLERANCE_V,
    }


@dataclass(frozen=True)
class GlobalGrid:
    """统一全局网格定义。

    默认参数严格按照需求：
    - `z in [0, 50 mm]`
    - `r in [0, 10 mm]`
    - `dz = dr = 0.1 mm`

    注意：这里全部转换到 SI 单位（米）后再存储。
    """

    z_min_m: float = 0.0
    z_max_m: float = 50.0e-3
    r_min_m: float = 0.0
    r_max_m: float = 10.0e-3
    dz_m: float = 0.1e-3
    dr_m: float = 0.1e-3
    capillary_exit_z_m: float = 5.0e-3

    def __post_init__(self) -> None:
        self._axis_node_count(
            axis_name="z",
            minimum_m=self.z_min_m,
            maximum_m=self.z_max_m,
            spacing_m=self.dz_m,
        )
        self._axis_node_count(
            axis_name="r",
            minimum_m=self.r_min_m,
            maximum_m=self.r_max_m,
            spacing_m=self.dr_m,
        )
        if not np.isfinite(self.capillary_exit_z_m):
            raise ValueError("capillary_exit_z_m must be finite.")

    @staticmethod
    def _axis_node_count(
        *,
        axis_name: str,
        minimum_m: float,
        maximum_m: float,
        spacing_m: float,
    ) -> int:
        if not all(np.isfinite(value) for value in (minimum_m, maximum_m, spacing_m)):
            raise ValueError(f"Global grid {axis_name} metadata must be finite.")
        if maximum_m <= minimum_m:
            raise ValueError(
                f"Global grid {axis_name}_max_m must exceed "
                f"{axis_name}_min_m."
            )
        if spacing_m <= 0.0:
            raise ValueError(f"Global grid d{axis_name}_m must be positive.")

        step_count_float = (maximum_m - minimum_m) / spacing_m
        step_count = int(round(step_count_float))
        if step_count < 1 or not np.isclose(
            step_count_float,
            step_count,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                f"Global grid {axis_name} extent must be an integer multiple "
                f"of d{axis_name}_m: extent={maximum_m - minimum_m}, "
                f"spacing={spacing_m}, ratio={step_count_float}."
            )
        return step_count + 1

    @classmethod
    def _axis_coordinates(
        cls,
        *,
        axis_name: str,
        minimum_m: float,
        maximum_m: float,
        spacing_m: float,
    ) -> np.ndarray:
        node_count = cls._axis_node_count(
            axis_name=axis_name,
            minimum_m=minimum_m,
            maximum_m=maximum_m,
            spacing_m=spacing_m,
        )
        return np.linspace(minimum_m, maximum_m, node_count, dtype=float)

    @property
    def z_coords_m(self) -> np.ndarray:
        return self._axis_coordinates(
            axis_name="z",
            minimum_m=self.z_min_m,
            maximum_m=self.z_max_m,
            spacing_m=self.dz_m,
        )

    @property
    def r_coords_m(self) -> np.ndarray:
        return self._axis_coordinates(
            axis_name="r",
            minimum_m=self.r_min_m,
            maximum_m=self.r_max_m,
            spacing_m=self.dr_m,
        )

    @property
    def mesh_rz(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回形状为 `(nr, nz)` 的二维网格。

        这里用 `x = z, y = r` 的 `xy` 约定，是为了后续画图时更自然：
        - 横坐标是 `z`
        - 纵坐标是 `r`

        返回：
        - `mesh_z_m`: 形状 `(nr, nz)`
        - `mesh_r_m`: 形状 `(nr, nz)`
        """

        mesh_z_m, mesh_r_m = np.meshgrid(self.z_coords_m, self.r_coords_m, indexing="xy")
        return mesh_r_m, mesh_z_m

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "z_min_m": self.z_min_m,
            "z_max_m": self.z_max_m,
            "r_min_m": self.r_min_m,
            "r_max_m": self.r_max_m,
            "dz_m": self.dz_m,
            "dr_m": self.dr_m,
            "z_coords_m": self.z_coords_m,
            "r_coords_m": self.r_coords_m,
            "capillary_exit_z_m": self.capillary_exit_z_m,
            "nr": len(self.r_coords_m),
            "nz": len(self.z_coords_m),
        }


@dataclass(frozen=True)
class FieldBakeConfig:
    """一次场烘焙任务的参数集合。"""

    simion_dc_csv: Optional[Path] = None
    simion_rf_csv: Optional[Path] = None
    fluent_csv: Optional[Path] = None
    offset_z_simion_mm: float = 0.0
    offset_z_fluent_mm: float = 5.0
    offset_z_fluent_mm_is_manual: bool = False
    force_offset_z_fluent_mm: bool = False
    output_npy: Path = Path("outputs/baked_fields.npy")
    output_plot_dir: Path = Path("outputs/field_baker_plots")
    background_pressure_pa: float = 200.0
    background_temperature_k: float = 300.0
    phi_key_for_plot: str = "phi_dc_v"
    z_min_mm: float = 0.0
    z_max_mm: float = 50.0
    r_min_mm: float = 0.0
    r_max_mm: float = 10.0
    dz_mm: float = 0.1
    dr_mm: float = 0.1
    capillary_exit_z_mm: float = 5.0
    clip_fluent_before_capillary_exit: bool = False
    clip_fluent_after_z_mm: Optional[float] = None
    fluent_capillary_total_length_mm: Optional[float] = None
    fluent_capillary_radius_mm: Optional[float] = None
    fluent_capillary_external_start_z_mm: Optional[float] = None
    simion_pa_effective_grids_per_mm: Optional[float] = None
    unsafe_allow_simion_pa_grid_mismatch: bool = False
    simion_dc_voltage_scale: float = 1.0

    def resolved_fluent_z_offset_mm(self) -> float:
        """Return the explicit Fluent z shift without deriving it from geometry."""

        return float(self.offset_z_fluent_mm)

    def fluent_z_offset_mode(self) -> str:
        if self.fluent_capillary_total_length_mm is None:
            return "manual_offset"
        return "manual_offset_with_capillary_metadata"

    def build_grid(self) -> GlobalGrid:
        """把 UI / CLI 中的毫米级网格设置转换成统一的 SI 网格。"""

        if self.z_max_mm <= self.z_min_mm:
            raise ValueError("z_max_mm 必须大于 z_min_mm。")
        if self.r_max_mm <= self.r_min_mm:
            raise ValueError("r_max_mm 必须大于 r_min_mm。")
        if self.dz_mm <= 0.0 or self.dr_mm <= 0.0:
            raise ValueError("dz_mm 和 dr_mm 必须为正数。")

        if (
            self.fluent_capillary_total_length_mm is not None
            and self.fluent_capillary_total_length_mm <= 0.0
        ):
            raise ValueError("--fluent-capillary-total-length-mm must be positive.")
        if self.fluent_capillary_radius_mm is not None and self.fluent_capillary_radius_mm <= 0.0:
            raise ValueError("--fluent-capillary-radius-mm must be positive.")
        if (
            self.fluent_capillary_external_start_z_mm is not None
            and self.fluent_capillary_external_start_z_mm >= self.capillary_exit_z_mm
        ):
            raise ValueError("--fluent-capillary-external-start-z-mm must be smaller than --capillary-exit-z-mm.")
        if self.clip_fluent_before_capillary_exit and (
            self.fluent_capillary_radius_mm is not None
            or self.fluent_capillary_external_start_z_mm is not None
        ):
            raise ValueError(
                "--clip-fluent-before-capillary-exit cannot be combined with the capillary external mask."
            )

        return GlobalGrid(
            z_min_m=self.z_min_mm * 1.0e-3,
            z_max_m=self.z_max_mm * 1.0e-3,
            r_min_m=self.r_min_mm * 1.0e-3,
            r_max_m=self.r_max_mm * 1.0e-3,
            dz_m=self.dz_mm * 1.0e-3,
            dr_m=self.dr_mm * 1.0e-3,
            capillary_exit_z_m=self.capillary_exit_z_mm * 1.0e-3,
        )


class FieldBaker:
    """统一的离线场烘焙器。"""

    def __init__(self, grid: Optional[GlobalGrid] = None) -> None:
        self.grid = grid or GlobalGrid()
        self._fields: Dict[str, Any] = {
            "grid": self.grid.to_metadata(),
            "simion": {},
            "fluent": {},
        }

    def _zero_simion_field(self, *, label: str) -> Dict[str, np.ndarray]:
        nr = len(self.grid.r_coords_m)
        nz = len(self.grid.z_coords_m)
        zeros = np.zeros((nr, nz), dtype=float)
        return {
            f"phi_{label}_v": zeros.copy(),
            f"e_{label}_r_v_per_m": zeros.copy(),
            f"e_{label}_z_v_per_m": zeros.copy(),
        }

    def _simion_regular_grid_to_matrix(
        self,
        r_unique_m: np.ndarray,
        z_unique_m: np.ndarray,
        r_values_m: np.ndarray,
        z_values_m: np.ndarray,
        potential_v: np.ndarray,
    ) -> np.ndarray:
        """把规整点表恢复成二维矩阵。

        输出矩阵形状是 `(nr_local, nz_local)`，也就是：
        - 第 0 维对应 `r`
        - 第 1 维对应 `z`
        """

        matrix = np.zeros((len(r_unique_m), len(z_unique_m)), dtype=float)
        counts = np.zeros_like(matrix, dtype=int)

        r_indices = np.searchsorted(r_unique_m, r_values_m)
        z_indices = np.searchsorted(z_unique_m, z_values_m)

        for i_r, i_z, value in zip(r_indices, z_indices, potential_v):
            matrix[i_r, i_z] += value
            counts[i_r, i_z] += 1

        mask = counts > 0
        matrix[mask] /= counts[mask]
        return matrix

    def _project_local_simion_field_to_global(
        self,
        local_phi_v: np.ndarray,
        local_r_coords_m: np.ndarray,
        local_z_coords_m: np.ndarray,
        *,
        z_offset_mm: float,
        label: str,
        source_format: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """把本地 SIMION 规则网格电势投影到统一全局网格。

        这一层负责统一做三件事：

        1. 给本地 `z` 坐标加上全局对齐偏移
        2. 用规则网格样条插值映射到全局 `(r, z)` 网格
        3. 用 `RectBivariateSpline` 解析导数求 `E_r / E_z`
        """

        z_offset_m = float(z_offset_mm) * 1.0e-3
        shifted_z_coords_m = local_z_coords_m + z_offset_m
        grid_r_m, grid_z_m = self.grid.mesh_rz

        spline_kx = min(3, len(local_r_coords_m) - 1)
        spline_ky = min(3, len(shifted_z_coords_m) - 1)
        spline = RectBivariateSpline(
            local_r_coords_m,
            shifted_z_coords_m,
            local_phi_v,
            kx=spline_kx,
            ky=spline_ky,
        )
        phi_global_v = spline(self.grid.r_coords_m, self.grid.z_coords_m)

        inside_mask = (
            (grid_r_m >= float(np.min(local_r_coords_m)))
            & (grid_r_m <= float(np.max(local_r_coords_m)))
            & (grid_z_m >= float(np.min(shifted_z_coords_m)))
            & (grid_z_m <= float(np.max(shifted_z_coords_m)))
        )
        phi_global_v = np.where(inside_mask, phi_global_v, 0.0)

        dphi_dr_v_per_m = spline(self.grid.r_coords_m, self.grid.z_coords_m, dx=1, dy=0)
        dphi_dz_v_per_m = spline(self.grid.r_coords_m, self.grid.z_coords_m, dx=0, dy=1)
        e_r_v_per_m = np.where(inside_mask, -dphi_dr_v_per_m, 0.0)
        e_z_v_per_m = np.where(inside_mask, -dphi_dz_v_per_m, 0.0)

        source_r_min_m = float(np.min(local_r_coords_m))
        source_r_max_m = float(np.max(local_r_coords_m))
        source_z_min_m = float(np.min(shifted_z_coords_m))
        source_z_max_m = float(np.max(shifted_z_coords_m))
        source_fully_inside = (
            source_r_min_m >= self.grid.r_min_m
            and source_r_max_m <= self.grid.r_max_m
            and source_z_min_m >= self.grid.z_min_m
            and source_z_max_m <= self.grid.z_max_m
        )

        baked: Dict[str, Any] = {
            f"phi_{label}_v": phi_global_v,
            f"e_{label}_r_v_per_m": e_r_v_per_m,
            f"e_{label}_z_v_per_m": e_z_v_per_m,
            f"{label}_z_offset_m": z_offset_m,
            f"{label}_source_format": source_format,
            f"{label}_source_r_min_m": source_r_min_m,
            f"{label}_source_r_max_m": source_r_max_m,
            f"{label}_source_z_min_m": source_z_min_m,
            f"{label}_source_z_max_m": source_z_max_m,
            f"{label}_source_fully_inside_global_grid": bool(source_fully_inside),
            f"{label}_electric_field_derivative_method": "rect_bivariate_spline_analytic_derivative",
            f"{label}_spline_degree_r": int(spline_kx),
            f"{label}_spline_degree_z": int(spline_ky),
        }
        if label == "rf":
            baked.update(_rf_reference_metadata())
        if extra_metadata:
            baked.update(extra_metadata)

        self._fields["simion"].update(baked)
        return baked

    def _bake_simion_csv_field(
        self,
        csv_path: Path,
        z_offset_mm: float,
        *,
        label: str,
        voltage_scale: float,
    ) -> Dict[str, np.ndarray]:
        """读取并烘焙一个传统 SIMION 电势 CSV。"""

        csv_path = Path(csv_path)
        columns = _read_csv_columns(csv_path)

        z_mm = _pick_column(columns, ("zmm", "z"), csv_path=csv_path)
        r_mm = _pick_column(columns, ("rmm", "r"), csv_path=csv_path)
        potential_v = _pick_column(columns, ("vv", "v", "potentialv", "voltagev"), csv_path=csv_path)

        z_values_m = z_mm * 1.0e-3
        r_values_m = r_mm * 1.0e-3

        valid_mask = np.isfinite(z_values_m) & np.isfinite(r_values_m) & np.isfinite(potential_v)
        z_values_m = z_values_m[valid_mask]
        r_values_m = r_values_m[valid_mask]
        potential_v = potential_v[valid_mask] * float(voltage_scale)

        r_unique_m = np.unique(r_values_m)
        z_unique_m = np.unique(z_values_m)
        expected_count = len(r_unique_m) * len(z_unique_m)

        if len(r_unique_m) >= 2 and len(z_unique_m) >= 2 and len(potential_v) == expected_count:
            local_phi_v = self._simion_regular_grid_to_matrix(
                r_unique_m,
                z_unique_m,
                r_values_m,
                z_values_m,
                potential_v,
            )
        else:
            mesh_z_local_m, mesh_r_local_m = np.meshgrid(z_unique_m, r_unique_m, indexing="xy")
            points = np.column_stack((r_values_m, z_values_m))
            local_phi_v = griddata(
                points,
                potential_v,
                (mesh_r_local_m, mesh_z_local_m),
                method="linear",
                fill_value=0.0,
            )

        return self._project_local_simion_field_to_global(
            local_phi_v,
            r_unique_m,
            z_unique_m,
            z_offset_mm=z_offset_mm,
            label=label,
            source_format="csv",
            extra_metadata={
                f"{label}_source_path": str(csv_path),
                f"{label}_csv_point_count": int(len(potential_v)),
                **({f"{label}_voltage_scale": float(voltage_scale)} if label != "rf" else {}),
            },
        )

    def _bake_simion_patxt_field(
        self,
        pa_path: Path,
        z_offset_mm: float,
        *,
        label: str,
        pa_effective_grids_per_mm: Optional[float],
        unsafe_allow_pa_grid_mismatch: bool,
        voltage_scale: float,
    ) -> Dict[str, np.ndarray]:
        """读取并烘焙一个 SIMION ASCII PA 文本文件。

        当前新工作流中，SIMION 已经在内部完成 Refine，SL Tool 导出的 `.patxt`
        保存的是全空间已知电势。因此这里不再做 Python 端 Laplace / SOR 求解，
        而是直接沿用原始电势。RF 文件必须已经是相邻 RF 电极 `+1 V / -1 V`
        的归一化基底场；运行时 RF 幅值只在主仿真中通过 `Vpeak / Vref` 进入。

        - 文件是 `xy` 平面
        - `symmetry = cylindrical`
        - 对物理坐标的解释应为 `x -> z`、`y -> r`
        """

        pa_path = Path(pa_path)
        header, pa_potential_v, electrode_mask = _read_simion_patxt(pa_path)

        if header["symmetry"].lower() != "cylindrical":
            raise ValueError(f"当前只支持 cylindrical SIMION PA 文本文件，但 {pa_path} 的 symmetry={header['symmetry']}")

        header_grids_per_mm = float(header["ng_from_header"])
        requested_grids_per_mm = (
            None
            if pa_effective_grids_per_mm is None
            else float(pa_effective_grids_per_mm)
        )
        if requested_grids_per_mm is None:
            effective_grids_per_mm = header_grids_per_mm
            grid_density_source = "patxt_header_ng"
        else:
            if (
                not np.isfinite(requested_grids_per_mm)
                or requested_grids_per_mm <= 0.0
            ):
                raise ValueError(
                    "simion_pa_effective_grids_per_mm must be finite and "
                    f"positive; got {requested_grids_per_mm} for {pa_path}."
                )
            density_matches_header = np.isclose(
                requested_grids_per_mm,
                header_grids_per_mm,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            if not density_matches_header and not unsafe_allow_pa_grid_mismatch:
                raise ValueError(
                    f"Requested SIMION PA grid density "
                    f"{requested_grids_per_mm:g} grids/mm disagrees with "
                    f"PATXT header ng={header_grids_per_mm:g} for {pa_path}. "
                    "Omit --simion-pa-grids-per-mm to infer ng from the "
                    "header, or use --unsafe-allow-simion-pa-grid-mismatch "
                    "only for an intentional coordinate reinterpretation."
                )
            effective_grids_per_mm = requested_grids_per_mm
            grid_density_source = (
                "explicit_matches_patxt_header_ng"
                if density_matches_header
                else "unsafe_explicit_override"
            )

        rf_validation_metadata: Dict[str, Any] = {}
        if label == "rf":
            rf_validation_metadata = _validate_rf_patxt_normalization(pa_path, pa_potential_v, electrode_mask)

        local_phi_v = pa_potential_v * float(voltage_scale)

        local_spacing_m = 1.0e-3 / effective_grids_per_mm
        # - 对 cylindrical SIMION `xy` 平面，当前采用：
        #   `x index -> axial z`
        #   `y index -> radial r`
        # - 因而本地数组形状 `(ny, nx)` 恰好对应 `(r, z)`。
        local_z_coords_m = np.arange(header["nx"], dtype=float) * local_spacing_m
        local_r_coords_m = np.arange(header["ny"], dtype=float) * local_spacing_m

        return self._project_local_simion_field_to_global(
            local_phi_v,
            local_r_coords_m,
            local_z_coords_m,
            z_offset_mm=z_offset_mm,
            label=label,
            source_format="patxt",
            extra_metadata={
                f"{label}_source_path": str(pa_path),
                f"{label}_pa_header_ng": float(header["ng_from_header"]),
                f"{label}_pa_requested_grids_per_mm": requested_grids_per_mm,
                f"{label}_pa_header_max_voltage": float(header["max_voltage"]),
                f"{label}_pa_field_type": str(header["field_type"]),
                f"{label}_pa_effective_grids_per_mm": float(effective_grids_per_mm),
                f"{label}_pa_grid_density_source": grid_density_source,
                f"{label}_pa_grid_density_mismatch_override": bool(
                    requested_grids_per_mm is not None
                    and not np.isclose(
                        requested_grids_per_mm,
                        header_grids_per_mm,
                        rtol=1.0e-12,
                        atol=1.0e-12,
                    )
                ),
                f"{label}_pa_coordinate_mapping": "x_index_to_global_z__y_index_to_global_r",
                **({f"{label}_voltage_scale": float(voltage_scale)} if label != "rf" else {}),
                f"{label}_pa_nx": int(header["nx"]),
                f"{label}_pa_ny": int(header["ny"]),
                f"{label}_pa_mirror_y": int(header["mirror_y"]),
                f"{label}_pa_point_count": int(header["point_count"]),
                f"{label}_pa_expected_point_count": int(
                    header["nx"] * header["ny"]
                ),
                f"{label}_pa_integrity_validated": bool(
                    header["integrity_validated"]
                ),
                f"{label}_pa_electrode_point_count": int(np.count_nonzero(electrode_mask)),
                f"{label}_pa_refined_in_simion": True,
                **rf_validation_metadata,
            },
        )

    def bake_simion_field(
        self,
        simion_path: Path,
        z_offset_mm: float,
        *,
        label: str,
        pa_effective_grids_per_mm: Optional[float],
        unsafe_allow_pa_grid_mismatch: bool = False,
        voltage_scale: float,
    ) -> Dict[str, np.ndarray]:
        """读取并烘焙一个 SIMION 电势文件。

        当前支持两类输入：

        - `.csv`：已经导出的规整 `(r, z, V)` 点表
        - `.patxt`：SIMION ASCII PA 文本阵列
        """

        if label == "rf" and not np.isclose(
            float(voltage_scale),
            RF_REFERENCE_PEAK_VOLTAGE_V,
            atol=RF_REFERENCE_NORMALIZATION_TOLERANCE_V,
            rtol=0.0,
        ):
            raise ValueError(
                "RF 烘焙不再接受 Vpp 或运行幅值缩放。"
                "请在 SIMION 中导出相邻 RF 电极 +1 V / -1 V 的归一化基底场；"
                "运行幅值只在主仿真中通过 --rf-peak-voltage 给出。"
            )

        simion_path = Path(simion_path)
        if simion_path.suffix.lower() == ".patxt":
            return self._bake_simion_patxt_field(
                simion_path,
                z_offset_mm,
                label=label,
                pa_effective_grids_per_mm=pa_effective_grids_per_mm,
                unsafe_allow_pa_grid_mismatch=unsafe_allow_pa_grid_mismatch,
                voltage_scale=voltage_scale,
            )
        return self._bake_simion_csv_field(simion_path, z_offset_mm, label=label, voltage_scale=voltage_scale)

    def bake_fluent_field(
        self,
        csv_path: Path,
        z_offset_mm: float,
        *,
        background_pressure_pa: float = 200.0,
        background_temperature_k: float = 300.0,
        background_vz_m_per_s: float = 0.0,
        background_vr_m_per_s: float = 0.0,
        clip_before_capillary_exit: bool = False,
        clip_after_z_mm: Optional[float] = None,
        capillary_total_length_mm: Optional[float] = None,
        capillary_radius_mm: Optional[float] = None,
        capillary_external_start_z_mm: Optional[float] = None,
        capillary_z_offset_mode: str = "manual_offset",
        capillary_nominal_z_offset_mm: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """读取并烘焙一个 Fluent 无序点云流场 CSV。

        输入 CSV 预期列：
        - `z(m)`
        - `r(m)`
        - `Pressure(Pa)`
        - `Temperature(K)`
        - `V_z(m/s)`
        - `V_r(m/s)`

        处理逻辑：
        1. 应用 `z_offset_mm` 把原始 Fluent 坐标对齐到统一全局 z
        2. 用 `scipy.interpolate.griddata(method='linear')` 将散点映射到统一网格
        3. 超出 Fluent 域的部分，用背景真空条件填充
        """

        csv_path = Path(csv_path)
        columns = _read_csv_columns(csv_path)

        raw_z_m = _pick_column(columns, ("zm", "z", "xcoordinate", "xcoordinatem"), csv_path=csv_path)
        r_m = _pick_column(columns, ("rm", "r", "ycoordinate", "ycoordinatem"), csv_path=csv_path)
        pressure_pa = _pick_column(
            columns,
            ("pressurepa", "ppa", "pressure", "absolutepressure", "staticpressure"),
            csv_path=csv_path,
        )
        temperature_k = _pick_column(
            columns,
            ("temperaturek", "tk", "temperature", "statictemperature"),
            csv_path=csv_path,
        )
        vz_m_per_s = _pick_column(
            columns,
            ("vzms", "velocityzms", "vzmps", "axialvelocity", "xvelocity", "velocityx"),
            csv_path=csv_path,
        )
        vr_m_per_s = _pick_column(
            columns,
            ("vrms", "velocityrms", "vrmps", "radialvelocity", "yvelocity", "velocityy"),
            csv_path=csv_path,
        )

        z_offset_m = float(z_offset_mm) * 1.0e-3
        z_m = raw_z_m + z_offset_m
        raw_point_count = int(z_m.size)
        capillary_exit_z_m = float(self.grid.capillary_exit_z_m)
        clip_after_z_m = None if clip_after_z_mm is None else float(clip_after_z_mm) * 1.0e-3
        capillary_total_length_m = (
            None if capillary_total_length_mm is None else float(capillary_total_length_mm) * 1.0e-3
        )
        capillary_radius_m = None if capillary_radius_mm is None else float(capillary_radius_mm) * 1.0e-3
        capillary_external_start_z_m = (
            None if capillary_external_start_z_mm is None else float(capillary_external_start_z_mm) * 1.0e-3
        )
        capillary_external_mask_enabled = (
            capillary_radius_m is not None and capillary_external_start_z_m is not None
        )
        if clip_before_capillary_exit and capillary_external_mask_enabled:
            raise ValueError(
                "clip_before_capillary_exit cannot be combined with capillary_radius_mm/"
                "capillary_external_start_z_mm."
            )
        clipped_point_count = int(np.count_nonzero(z_m < capillary_exit_z_m)) if clip_before_capillary_exit else 0
        clipped_after_point_count = int(np.count_nonzero(z_m > clip_after_z_m)) if clip_after_z_m is not None else 0

        valid_mask = (
            np.isfinite(z_m)
            & np.isfinite(r_m)
            & np.isfinite(pressure_pa)
            & np.isfinite(temperature_k)
            & np.isfinite(vz_m_per_s)
            & np.isfinite(vr_m_per_s)
        )
        finite_point_count = int(np.count_nonzero(valid_mask))
        capillary_masked_source_point_count = 0
        capillary_external_source_point_count = 0
        if capillary_external_mask_enabled:
            assert capillary_external_start_z_m is not None
            assert capillary_radius_m is not None
            external_source_mask = (z_m >= capillary_exit_z_m) | (
                (z_m >= capillary_external_start_z_m)
                & (z_m < capillary_exit_z_m)
                & (r_m > capillary_radius_m)
            )
            capillary_external_source_point_count = int(np.count_nonzero(valid_mask & external_source_mask))
            capillary_masked_source_point_count = int(np.count_nonzero(valid_mask & ~external_source_mask))
            valid_mask = valid_mask & external_source_mask

        z_m = z_m[valid_mask]
        raw_z_m = raw_z_m[valid_mask]
        r_m = r_m[valid_mask]
        pressure_pa = pressure_pa[valid_mask]
        temperature_k = temperature_k[valid_mask]
        vz_m_per_s = vz_m_per_s[valid_mask]
        vr_m_per_s = vr_m_per_s[valid_mask]
        if z_m.size == 0:
            raise ValueError(
                "No valid Fluent points remain after filtering. "
                "Check --offset-z-fluent-mm and Fluent clipping options."
            )

        points = np.column_stack((r_m, z_m))
        grid_r_m, grid_z_m = self.grid.mesh_rz

        # - `griddata(linear)` 适合 Fluent 导出的无序散点：
        #   先在散点形成的凸包内做线性单纯形插值。
        # - 超出 Fluent 域的区域，不延拓真实流场，而是明确填成背景真空条件，
        #   这样主程序在边界外不会读到虚假的高压或高速。
        pressure_global_pa = griddata(
            points,
            pressure_pa,
            (grid_r_m, grid_z_m),
            method="linear",
            fill_value=float(background_pressure_pa),
        )
        temperature_global_k = griddata(
            points,
            temperature_k,
            (grid_r_m, grid_z_m),
            method="linear",
            fill_value=float(background_temperature_k),
        )
        vz_global_m_per_s = griddata(
            points,
            vz_m_per_s,
            (grid_r_m, grid_z_m),
            method="linear",
            fill_value=float(background_vz_m_per_s),
        )
        vr_global_m_per_s = griddata(
            points,
            vr_m_per_s,
            (grid_r_m, grid_z_m),
            method="linear",
            fill_value=float(background_vr_m_per_s),
        )
        capillary_masked_grid_cell_count = 0
        if capillary_external_mask_enabled:
            assert capillary_external_start_z_m is not None
            assert capillary_radius_m is not None
            external_grid_mask = (grid_z_m >= capillary_exit_z_m) | (
                (grid_z_m >= capillary_external_start_z_m)
                & (grid_z_m < capillary_exit_z_m)
                & (grid_r_m > capillary_radius_m)
            )
            capillary_masked_grid_cell_count = int(np.count_nonzero(~external_grid_mask))
            pressure_global_pa = np.where(external_grid_mask, pressure_global_pa, float(background_pressure_pa))
            temperature_global_k = np.where(external_grid_mask, temperature_global_k, float(background_temperature_k))
            vz_global_m_per_s = np.where(external_grid_mask, vz_global_m_per_s, float(background_vz_m_per_s))
            vr_global_m_per_s = np.where(external_grid_mask, vr_global_m_per_s, float(background_vr_m_per_s))

        if clip_before_capillary_exit:
            before_exit_mask = grid_z_m < capillary_exit_z_m
            pressure_global_pa = np.where(before_exit_mask, float(background_pressure_pa), pressure_global_pa)
            temperature_global_k = np.where(before_exit_mask, float(background_temperature_k), temperature_global_k)
            vz_global_m_per_s = np.where(before_exit_mask, float(background_vz_m_per_s), vz_global_m_per_s)
            vr_global_m_per_s = np.where(before_exit_mask, float(background_vr_m_per_s), vr_global_m_per_s)

        if clip_after_z_m is not None:
            after_clip_mask = grid_z_m > clip_after_z_m
            pressure_global_pa = np.where(after_clip_mask, float(background_pressure_pa), pressure_global_pa)
            temperature_global_k = np.where(after_clip_mask, float(background_temperature_k), temperature_global_k)
            vz_global_m_per_s = np.where(after_clip_mask, float(background_vz_m_per_s), vz_global_m_per_s)
            vr_global_m_per_s = np.where(after_clip_mask, float(background_vr_m_per_s), vr_global_m_per_s)

        source_velocity_raw_z_min_m = None
        source_velocity_raw_z_max_m = None
        if capillary_total_length_m is not None:
            source_velocity_raw_z_min_m = float(self.grid.z_min_m - z_offset_m)
            source_velocity_raw_z_max_m = float(capillary_exit_z_m - z_offset_m)

        baked = {
            "pressure_pa": pressure_global_pa,
            "temperature_k": temperature_global_k,
            "v_z_m_per_s": vz_global_m_per_s,
            "v_r_m_per_s": vr_global_m_per_s,
            "fluent_z_offset_m": z_offset_m,
            "fluent_z_offset_mode": str(capillary_z_offset_mode),
            "fluent_nominal_capillary_z_offset_m": (
                None if capillary_nominal_z_offset_mm is None else float(capillary_nominal_z_offset_mm) * 1.0e-3
            ),
            "fluent_source_path": str(csv_path),
            "fluent_point_count": int(len(z_m)),
            "fluent_raw_point_count": raw_point_count,
            "fluent_finite_point_count": finite_point_count,
            "fluent_clipped_before_capillary_exit": bool(clip_before_capillary_exit),
            "fluent_clip_z_min_m": capillary_exit_z_m if clip_before_capillary_exit else None,
            "fluent_clipped_point_count": clipped_point_count,
            "fluent_clipped_after_z": clip_after_z_m is not None,
            "fluent_clip_z_max_m": clip_after_z_m,
            "fluent_clipped_after_point_count": clipped_after_point_count,
            "fluent_capillary_total_length_m": capillary_total_length_m,
            "fluent_capillary_radius_m": capillary_radius_m,
            "fluent_capillary_exit_z_m": capillary_exit_z_m,
            "fluent_capillary_external_start_z_m": capillary_external_start_z_m,
            "fluent_capillary_external_mask_enabled": bool(capillary_external_mask_enabled),
            "fluent_capillary_masked_source_point_count": capillary_masked_source_point_count,
            "fluent_capillary_external_source_point_count": capillary_external_source_point_count,
            "fluent_capillary_masked_grid_cell_count": capillary_masked_grid_cell_count,
            "fluent_capillary_source_velocity_raw_z_min_m": source_velocity_raw_z_min_m,
            "fluent_capillary_source_velocity_raw_z_max_m": source_velocity_raw_z_max_m,
            "fluent_capillary_source_velocity_global_z_min_m": (
                float(self.grid.z_min_m) if capillary_total_length_m is not None else None
            ),
            "fluent_capillary_source_velocity_global_z_max_m": (
                capillary_exit_z_m if capillary_total_length_m is not None else None
            ),
            "fluent_coordinate_mapping": "x_coordinate_to_global_z__y_coordinate_to_global_r",
            "fluent_raw_source_z_min_m": float(np.min(raw_z_m)) if raw_z_m.size else float("nan"),
            "fluent_raw_source_z_max_m": float(np.max(raw_z_m)) if raw_z_m.size else float("nan"),
            "fluent_source_z_min_m": float(np.min(z_m)) if z_m.size else float("nan"),
            "fluent_source_z_max_m": float(np.max(z_m)) if z_m.size else float("nan"),
            "fluent_source_r_min_m": float(np.min(r_m)) if r_m.size else float("nan"),
            "fluent_source_r_max_m": float(np.max(r_m)) if r_m.size else float("nan"),
            "background_pressure_pa": float(background_pressure_pa),
            "background_temperature_k": float(background_temperature_k),
            "background_vz_m_per_s": float(background_vz_m_per_s),
            "background_vr_m_per_s": float(background_vr_m_per_s),
        }
        self._fields["fluent"].update(baked)
        return baked

    def export_to_npy(self, output_path: Path) -> Path:
        """导出统一烘焙结果为 `.npy`。

        读回方式：

        ```python
        baked = np.load("baked_fields.npy", allow_pickle=True).item()
        ```
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, self._fields, allow_pickle=True)
        return output_path

    def plot_sanity_check(self, output_dir: Path, *, phi_key: str = "phi_dc_v") -> Sequence[Path]:
        """生成诊断图，检查毛细管出口是否对齐正确。

        图像内容：
        1. 电势热力图
        2. 压强热力图
        3. 轴线 `r = 0` 上的一维参数变化曲线
        """

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        z_mm = self.grid.z_coords_m * 1.0e3
        r_mm = self.grid.r_coords_m * 1.0e3
        extent = [z_mm[0], z_mm[-1], r_mm[0], r_mm[-1]]

        phi_v = self._fields["simion"].get(phi_key)
        if phi_v is None:
            if "phi_rf_v" in self._fields["simion"]:
                phi_v = self._fields["simion"]["phi_rf_v"]
                phi_key = "phi_rf_v"
            else:
                phi_v = np.zeros((len(r_mm), len(z_mm)), dtype=float)

        pressure_pa = self._fields["fluent"].get("pressure_pa", np.zeros((len(r_mm), len(z_mm)), dtype=float))
        temperature_k = self._fields["fluent"].get("temperature_k", np.zeros((len(r_mm), len(z_mm)), dtype=float))
        vz_m_per_s = self._fields["fluent"].get("v_z_m_per_s", np.zeros((len(r_mm), len(z_mm)), dtype=float))
        vr_m_per_s = self._fields["fluent"].get("v_r_m_per_s", np.zeros((len(r_mm), len(z_mm)), dtype=float))

        capillary_exit_z_mm = self.grid.capillary_exit_z_m * 1.0e3
        axis_index = 0

        heatmap_path = output_dir / "sanity_phi_pressure_heatmaps.png"
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=160)

        im0 = axes[0].imshow(phi_v, origin="lower", aspect="auto", extent=extent, cmap="turbo")
        axes[0].axvline(
            capillary_exit_z_mm,
            color="white",
            linestyle="--",
            linewidth=1.5,
            label=f"Capillary Exit ({capillary_exit_z_mm:g} mm)",
        )
        axes[0].set_title(f"Interpolated Potential Heatmap: {phi_key}")
        axes[0].set_xlabel("z [mm]")
        axes[0].set_ylabel("r [mm]")
        axes[0].legend(loc="upper right")
        fig.colorbar(im0, ax=axes[0], label="Potential [V]")

        im1 = axes[1].imshow(pressure_pa, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        axes[1].axvline(
            capillary_exit_z_mm,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Capillary Exit ({capillary_exit_z_mm:g} mm)",
        )
        axes[1].set_title("Interpolated Pressure Heatmap")
        axes[1].set_xlabel("z [mm]")
        axes[1].set_ylabel("r [mm]")
        axes[1].legend(loc="upper right")
        fig.colorbar(im1, ax=axes[1], label="Pressure [Pa]")

        fig.tight_layout()
        fig.savefig(heatmap_path)
        plt.close(fig)

        centerline_path = output_dir / "sanity_centerline_profiles.png"
        fig, axes = plt.subplots(3, 1, figsize=(10.0, 9.0), dpi=160, sharex=True)

        axes[0].plot(z_mm, phi_v[axis_index], label="Potential [V]", color="#1f77b4")
        axes[0].plot(z_mm, pressure_pa[axis_index], label="Pressure [Pa]", color="#d62728")
        axes[0].axvline(capillary_exit_z_mm, color="black", linestyle="--", linewidth=1.2)
        axes[0].set_ylabel("Potential / Pressure")
        axes[0].set_title("Centerline Profiles at r = 0")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        axes[1].plot(z_mm, temperature_k[axis_index], label="Temperature [K]", color="#ff7f0e")
        axes[1].axvline(capillary_exit_z_mm, color="black", linestyle="--", linewidth=1.2)
        axes[1].set_ylabel("Temperature [K]")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend()

        axes[2].plot(z_mm, vz_m_per_s[axis_index], label="Vz [m/s]", color="#2ca02c")
        axes[2].plot(z_mm, vr_m_per_s[axis_index], label="Vr [m/s]", color="#9467bd")
        axes[2].axvline(capillary_exit_z_mm, color="black", linestyle="--", linewidth=1.2)
        axes[2].set_xlabel("z [mm]")
        axes[2].set_ylabel("Velocity [m/s]")
        axes[2].grid(True, alpha=0.25)
        axes[2].legend()

        fig.tight_layout()
        fig.savefig(centerline_path)
        plt.close(fig)

        return [heatmap_path, centerline_path]


def run_field_bake(config: FieldBakeConfig, *, grid: Optional[GlobalGrid] = None) -> Dict[str, Any]:
    """执行一次完整的离线烘焙流程。

    这个函数把“读取 -> 对齐 -> 插值 -> 导出 -> 诊断出图”封装成一个统一入口，
    方便命令行和 Tkinter UI 复用同一套逻辑。
    """

    resolved_grid = grid or config.build_grid()
    baker = FieldBaker(grid=resolved_grid)

    if config.simion_dc_csv:
        baker.bake_simion_field(
            Path(config.simion_dc_csv),
            config.offset_z_simion_mm,
            label="dc",
            pa_effective_grids_per_mm=config.simion_pa_effective_grids_per_mm,
            unsafe_allow_pa_grid_mismatch=config.unsafe_allow_simion_pa_grid_mismatch,
            voltage_scale=config.simion_dc_voltage_scale,
        )
    else:
        baker._fields["simion"].update(baker._zero_simion_field(label="dc"))

    if config.simion_rf_csv:
        baker.bake_simion_field(
            Path(config.simion_rf_csv),
            config.offset_z_simion_mm,
            label="rf",
            pa_effective_grids_per_mm=config.simion_pa_effective_grids_per_mm,
            unsafe_allow_pa_grid_mismatch=config.unsafe_allow_simion_pa_grid_mismatch,
            voltage_scale=RF_REFERENCE_PEAK_VOLTAGE_V,
        )
    else:
        baker._fields["simion"].update(baker._zero_simion_field(label="rf"))

    if config.fluent_csv:
        fluent_z_offset_mm = config.resolved_fluent_z_offset_mm()
        nominal_capillary_offset_mm = (
            None
            if config.fluent_capillary_total_length_mm is None
            else float(config.capillary_exit_z_mm) - float(config.fluent_capillary_total_length_mm)
        )
        baker.bake_fluent_field(
            Path(config.fluent_csv),
            fluent_z_offset_mm,
            background_pressure_pa=config.background_pressure_pa,
            background_temperature_k=config.background_temperature_k,
            clip_before_capillary_exit=config.clip_fluent_before_capillary_exit,
            clip_after_z_mm=config.clip_fluent_after_z_mm,
            capillary_total_length_mm=config.fluent_capillary_total_length_mm,
            capillary_radius_mm=config.fluent_capillary_radius_mm,
            capillary_external_start_z_mm=config.fluent_capillary_external_start_z_mm,
            capillary_z_offset_mode=config.fluent_z_offset_mode(),
            capillary_nominal_z_offset_mm=nominal_capillary_offset_mm,
        )
    else:
        grid_r_m, grid_z_m = baker.grid.mesh_rz
        baker._fields["fluent"].update(
            {
                "pressure_pa": np.full_like(grid_r_m, config.background_pressure_pa, dtype=float),
                "temperature_k": np.full_like(grid_r_m, config.background_temperature_k, dtype=float),
                "v_z_m_per_s": np.zeros_like(grid_r_m, dtype=float),
                "v_r_m_per_s": np.zeros_like(grid_r_m, dtype=float),
                "fluent_z_offset_m": float(config.resolved_fluent_z_offset_mm()) * 1.0e-3,
                "fluent_z_offset_mode": str(config.fluent_z_offset_mode()),
                "background_pressure_pa": float(config.background_pressure_pa),
                "background_temperature_k": float(config.background_temperature_k),
                "background_vz_m_per_s": 0.0,
                "background_vr_m_per_s": 0.0,
            }
        )

    output_npy = baker.export_to_npy(Path(config.output_npy))
    plot_paths = baker.plot_sanity_check(Path(config.output_plot_dir), phi_key=config.phi_key_for_plot)
    return {
        "output_npy": output_npy,
        "plot_paths": list(plot_paths),
        "field_keys": sorted(baker._fields.keys()),
        "grid_metadata": baker.grid.to_metadata(),
        "simion_summary": {key: value for key, value in baker._fields["simion"].items() if not isinstance(value, np.ndarray)},
        "baker": baker,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线烘焙 SIMION / Fluent 场到统一 2D 轴对称网格。")
    parser.add_argument("--simion-dc-csv", default="", help="SIMION DC 电势文件路径，可为 CSV 或 .patxt。")
    parser.add_argument("--simion-rf-csv", default="", help="SIMION RF 归一化基底电势文件路径，可为 CSV 或 .patxt；RF 电极必须为 +1 V / -1 V。")
    parser.add_argument("--fluent-csv", default="", help="Fluent 流场 CSV 路径。")
    parser.add_argument("--offset-z-simion-mm", type=float, default=0.0, help="SIMION z 轴全局对齐偏移量，单位 mm。")
    parser.add_argument("--offset-z-fluent-mm", type=float, default=5.0, help="Fluent z 轴全局对齐偏移量，单位 mm。")
    parser.add_argument("--output-npy", default="outputs/baked_fields.npy", help="导出的 `.npy` 文件路径。")
    parser.add_argument("--output-plot-dir", default="outputs/field_baker_plots", help="诊断图输出目录。")
    parser.add_argument("--background-pressure-pa", type=float, default=200.0, help="Fluent 域外背景压强 [Pa]。")
    parser.add_argument("--background-temperature-k", type=float, default=300.0, help="Fluent 域外背景温度 [K]。")
    parser.add_argument("--phi-key-for-plot", default="phi_dc_v", choices=("phi_dc_v", "phi_rf_v"), help="诊断热力图使用的电势字段。")
    parser.add_argument("--z-min-mm", type=float, default=0.0, help="全局网格 z 最小值 [mm]。")
    parser.add_argument("--z-max-mm", type=float, default=50.0, help="全局网格 z 最大值 [mm]。")
    parser.add_argument("--r-min-mm", type=float, default=0.0, help="全局网格 r 最小值 [mm]。")
    parser.add_argument("--r-max-mm", type=float, default=10.0, help="全局网格 r 最大值 [mm]。")
    parser.add_argument("--dz-mm", type=float, default=0.1, help="全局网格 z 步长 [mm]。")
    parser.add_argument("--dr-mm", type=float, default=0.1, help="全局网格 r 步长 [mm]。")
    parser.add_argument(
        "--capillary-exit-z-mm",
        type=float,
        default=5.0,
        help="全局坐标中的 capillary 出口位置 [mm]，写入 baked grid metadata 并用于诊断图。",
    )
    parser.add_argument(
        "--clip-fluent-before-capillary-exit",
        action="store_true",
        help="丢弃全局 z 小于 capillary exit 的 Fluent 点；用于 Fluent 文件包含 capillary 内部流场但外部模拟从出口开始的情况。",
    )
    parser.add_argument(
        "--simion-pa-grids-per-mm",
        type=float,
        default=None,
        help=(
            "Optional SIMION `.patxt` physical grids/mm. By default this is "
            "inferred from header ng; an explicit value must match the header."
        ),
    )
    parser.add_argument(
        "--unsafe-allow-simion-pa-grid-mismatch",
        action="store_true",
        help=(
            "Allow --simion-pa-grids-per-mm to disagree with PATXT header ng. "
            "This intentionally reinterprets geometry and is unsafe."
        ),
    )
    parser.add_argument("--simion-dc-voltage-scale", type=float, default=1.0, help="DC 电势文件的电压缩放系数；若文件已是真实 DC 电压，设为 1。")
    parser.add_argument(
        "--clip-fluent-after-z-mm",
        type=float,
        default=None,
        help="Set Fluent gas values to background for global z greater than this outlet/end-plane position [mm].",
    )
    parser.add_argument(
        "--simion-rf-voltage-scale",
        "--simion-rf-vpp",
        dest="deprecated_simion_rf_voltage_scale",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--force-offset-z-fluent-mm",
        action="store_true",
        help="Deprecated compatibility flag; Fluent z offset is always used directly.",
    )
    parser.add_argument(
        "--fluent-capillary-total-length-mm",
        type=float,
        default=None,
        help="Physical capillary length metadata; it does not modify --offset-z-fluent-mm.",
    )
    parser.add_argument(
        "--fluent-capillary-radius-mm",
        type=float,
        default=None,
        help="Capillary radius used to mask capillary-interior gas from the baked external Fluent field [mm].",
    )
    parser.add_argument(
        "--fluent-capillary-external-start-z-mm",
        type=float,
        default=None,
        help="Global z where the capillary outer surface enters the external S-lens gas domain [mm].",
    )
    args = parser.parse_args()
    args.offset_z_fluent_mm_is_manual = any(
        token == "--offset-z-fluent-mm" or token.startswith("--offset-z-fluent-mm=")
        for token in sys.argv[1:]
    )
    if args.deprecated_simion_rf_voltage_scale is not None:
        parser.error(
            "RF 烘焙已固定为 Vref=1 V 的 +1 V / -1 V 归一化基底场；"
            "不要传 --simion-rf-vpp 或 --simion-rf-voltage-scale。"
            "运行 RF 峰值请在主仿真中使用 --rf-peak-voltage。"
        )
    return args


def main() -> None:
    args = parse_args()
    result = run_field_bake(
        FieldBakeConfig(
            simion_dc_csv=Path(args.simion_dc_csv) if args.simion_dc_csv else None,
            simion_rf_csv=Path(args.simion_rf_csv) if args.simion_rf_csv else None,
            fluent_csv=Path(args.fluent_csv) if args.fluent_csv else None,
            offset_z_simion_mm=args.offset_z_simion_mm,
            offset_z_fluent_mm=args.offset_z_fluent_mm,
            offset_z_fluent_mm_is_manual=bool(args.offset_z_fluent_mm_is_manual),
            force_offset_z_fluent_mm=bool(args.force_offset_z_fluent_mm),
            output_npy=Path(args.output_npy),
            output_plot_dir=Path(args.output_plot_dir),
            background_pressure_pa=args.background_pressure_pa,
            background_temperature_k=args.background_temperature_k,
            phi_key_for_plot=args.phi_key_for_plot,
            z_min_mm=args.z_min_mm,
            z_max_mm=args.z_max_mm,
            r_min_mm=args.r_min_mm,
            r_max_mm=args.r_max_mm,
            dz_mm=args.dz_mm,
            dr_mm=args.dr_mm,
            capillary_exit_z_mm=args.capillary_exit_z_mm,
            clip_fluent_before_capillary_exit=bool(args.clip_fluent_before_capillary_exit),
            clip_fluent_after_z_mm=args.clip_fluent_after_z_mm,
            fluent_capillary_total_length_mm=args.fluent_capillary_total_length_mm,
            fluent_capillary_radius_mm=args.fluent_capillary_radius_mm,
            fluent_capillary_external_start_z_mm=args.fluent_capillary_external_start_z_mm,
            simion_pa_effective_grids_per_mm=args.simion_pa_grids_per_mm,
            unsafe_allow_simion_pa_grid_mismatch=bool(
                args.unsafe_allow_simion_pa_grid_mismatch
            ),
            simion_dc_voltage_scale=args.simion_dc_voltage_scale,
        )
    )

    grid_metadata = result["grid_metadata"]
    print(
        "Resolved global grid: "
        f"r=[{grid_metadata['r_min_m'] * 1.0e3:.3f}, {grid_metadata['r_max_m'] * 1.0e3:.3f}] mm, "
        f"z=[{grid_metadata['z_min_m'] * 1.0e3:.3f}, {grid_metadata['z_max_m'] * 1.0e3:.3f}] mm, "
        f"dr={grid_metadata['dr_m'] * 1.0e3:.3f} mm, dz={grid_metadata['dz_m'] * 1.0e3:.3f} mm, "
        f"capillary_exit_z={grid_metadata['capillary_exit_z_m'] * 1.0e3:.3f} mm, "
        f"shape=({grid_metadata['nr']}, {grid_metadata['nz']})"
    )
    print(f"Saved baked field file: {result['output_npy']}")
    for path in result["plot_paths"]:
        print(f"Saved sanity plot: {path}")


if __name__ == "__main__":
    main()
