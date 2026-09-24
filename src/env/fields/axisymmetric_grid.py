"""Environment-owned axisymmetric field buffers for the unified 2D (r, z) grid.

The particles live in 3D Cartesian coordinates ``(x, y, z)``, but all Eulerian
fields are stored on a 2D axisymmetric mesh. Particle-to-grid and grid-to-particle
operators therefore always map through the cylindrical radius

``r = sqrt(x^2 + y^2)``.
"""

import math
from typing import Any, Optional

import taichi as ti


STATIC_RUNTIME_FIELD_NAMES = (
    "E_dc_r",
    "E_dc_z",
    "E_rf_r",
    "E_rf_z",
    "P_gas",
    "T_gas",
    "v_gas_r",
    "v_gas_z",
)
STATIC_POTENTIAL_FIELD_NAMES = ("phi_dc", "phi_rf")
PIC_FIELD_NAMES = ("rho_charge", "phi_sce", "E_sce_r", "E_sce_z")
STATIC_CACHE_FIELD_NAMES = STATIC_RUNTIME_FIELD_NAMES
PIC_CACHE_FIELD_NAMES = ("E_sce_r", "E_sce_z")
ALL_2D_FIELD_NAMES = STATIC_POTENTIAL_FIELD_NAMES + STATIC_RUNTIME_FIELD_NAMES + PIC_FIELD_NAMES

_STORAGE_MODE_FIELD_NAMES = {
    "unified": ALL_2D_FIELD_NAMES,
    "static": STATIC_RUNTIME_FIELD_NAMES,
    "pic": PIC_FIELD_NAMES,
}
_FLOAT64_BYTES = 8


def resolve_pic_grid_shape(
    *,
    static_nr: int,
    static_nz: int,
    r_max_m: float,
    z_min_m: float,
    z_max_m: float,
    requested_nr: Optional[int],
    requested_nz: Optional[int],
) -> tuple[int, int, str]:
    """Resolve default-shared, explicitly shared, or decoupled PIC semantics.

    ``None/None`` shares the static grid by default. ``0/0`` explicitly selects
    the same shared mode. A positive pair specifies exact PIC node counts; the
    runtime reuses the unified grid when that shape equals the static shape.
    """

    static_nr = int(static_nr)
    static_nz = int(static_nz)
    if static_nr < 3 or static_nz < 3:
        raise ValueError("The static field grid requires at least 3x3 nodes.")
    if r_max_m <= 0.0 or z_max_m <= z_min_m:
        raise ValueError("Invalid axisymmetric domain extents.")

    if requested_nr is None and requested_nz is None:
        return static_nr, static_nz, "default_shared_static"

    if requested_nr is None or requested_nz is None:
        raise ValueError("pic_grid_nr and pic_grid_nz must be specified together.")

    requested_nr = int(requested_nr)
    requested_nz = int(requested_nz)
    if requested_nr == 0 and requested_nz == 0:
        return static_nr, static_nz, "shared_static"
    if requested_nr < 3 or requested_nz < 3:
        raise ValueError(
            "PIC grid node counts must both be 0 (share static) or both be at least 3."
        )
    return requested_nr, requested_nz, "explicit"


def estimate_grid_storage_bytes(nr: int, nz: int, *, storage_mode: str = "unified") -> int:
    """Estimate dense float64 field payload plus the two coordinate arrays."""

    if storage_mode not in _STORAGE_MODE_FIELD_NAMES:
        raise ValueError(f"Unsupported UnifiedGrid2D storage mode: {storage_mode!r}.")
    nr = int(nr)
    nz = int(nz)
    if nr < 3 or nz < 3:
        raise ValueError("Grid storage estimates require at least 3x3 nodes.")
    field_count = len(_STORAGE_MODE_FIELD_NAMES[storage_mode])
    return _FLOAT64_BYTES * (field_count * nr * nz + nr + nz)


def estimate_runtime_grid_resources(
    *,
    static_nr: int,
    static_nz: int,
    pic_nr: int,
    pic_nz: int,
) -> dict[str, Any]:
    """Return steady dense-grid/cache payload estimates for the coupled runtime."""

    static_nr = int(static_nr)
    static_nz = int(static_nz)
    pic_nr = int(pic_nr)
    pic_nz = int(pic_nz)
    static_nodes = static_nr * static_nz
    pic_nodes = pic_nr * pic_nz
    shared = static_nr == pic_nr and static_nz == pic_nz
    static_mode = "unified" if shared else "static"
    static_grid_bytes = estimate_grid_storage_bytes(static_nr, static_nz, storage_mode=static_mode)
    pic_grid_bytes = 0 if shared else estimate_grid_storage_bytes(pic_nr, pic_nz, storage_mode="pic")
    static_cache_bytes = _FLOAT64_BYTES * len(STATIC_CACHE_FIELD_NAMES) * static_nodes
    pic_cache_bytes = _FLOAT64_BYTES * len(PIC_CACHE_FIELD_NAMES) * pic_nodes
    total_bytes = static_grid_bytes + pic_grid_bytes + static_cache_bytes + pic_cache_bytes
    return {
        "static_nodes": static_nodes,
        "pic_nodes": pic_nodes,
        "shared_grid": shared,
        "static_storage_mode": static_mode,
        "pic_storage_mode": "shared" if shared else "pic",
        "static_grid_bytes": static_grid_bytes,
        "pic_grid_bytes": pic_grid_bytes,
        "static_cache_bytes": static_cache_bytes,
        "pic_cache_bytes": pic_cache_bytes,
        "grid_bytes": static_grid_bytes + pic_grid_bytes,
        "cache_bytes": static_cache_bytes + pic_cache_bytes,
        "total_bytes": total_bytes,
    }


@ti.data_oriented
class UnifiedGrid2D:
    """Unified 2D axisymmetric field container.

    Parameters
    ----------
    nr, nz
        Number of grid nodes in radial and axial directions.
    r_max_m
        Maximum radius of the axisymmetric domain in meters.
    z_min_m, z_max_m
        Axial bounds of the domain in meters.

    Notes
    -----
    All fields use SI units:

    - potential: ``V``
    - electric field: ``V/m``
    - pressure: ``Pa``
    - temperature: ``K``
    - gas velocity: ``m/s``
    - charge density: ``C/m^3``
    """

    def __init__(
        self,
        nr: int,
        nz: int,
        r_max_m: float,
        z_min_m: float,
        z_max_m: float,
        *,
        storage_mode: str = "unified",
    ) -> None:
        if nr < 3 or nz < 3:
            raise ValueError("UnifiedGrid2D requires at least 3x3 nodes.")
        if r_max_m <= 0.0 or z_max_m <= z_min_m:
            raise ValueError("Invalid axisymmetric domain extents.")
        if storage_mode not in _STORAGE_MODE_FIELD_NAMES:
            raise ValueError(f"Unsupported UnifiedGrid2D storage mode: {storage_mode!r}.")

        self.nr = int(nr)
        self.nz = int(nz)
        self.r_max_m = float(r_max_m)
        self.z_min_m = float(z_min_m)
        self.z_max_m = float(z_max_m)
        self.dr = self.r_max_m / float(self.nr - 1)
        self.dz = (self.z_max_m - self.z_min_m) / float(self.nz - 1)
        self.storage_mode = storage_mode
        self.has_static_fields = storage_mode in {"unified", "static"}
        self.has_potential_fields = storage_mode == "unified"
        self.has_pic_fields = storage_mode in {"unified", "pic"}
        self.allocated_2d_field_names = _STORAGE_MODE_FIELD_NAMES[storage_mode]
        self.estimated_storage_bytes = estimate_grid_storage_bytes(
            self.nr,
            self.nz,
            storage_mode=storage_mode,
        )

        self.r = ti.field(dtype=ti.f64, shape=self.nr)
        self.z = ti.field(dtype=ti.f64, shape=self.nz)

        # Missing-role attributes deliberately remain visible as None.  This
        # makes accidental static/PIC cross-use fail clearly without allocating
        # multi-gigabyte compatibility buffers on decoupled production meshes.
        for field_name in ALL_2D_FIELD_NAMES:
            setattr(self, field_name, None)
        for field_name in self.allocated_2d_field_names:
            setattr(self, field_name, ti.field(dtype=ti.f64, shape=(self.nr, self.nz)))

        self._initialize_coordinates()
        if self.has_pic_fields:
            self.clear_charge()
            self.clear_space_charge_solution()

    @ti.kernel
    def _initialize_coordinates(self):
        """Fill the 1D coordinate arrays for diagnostic use."""

        for i in self.r:
            self.r[i] = float(i) * self.dr
        for k in self.z:
            self.z[k] = self.z_min_m + float(k) * self.dz

    @ti.func
    def clamp_r_index(self, i: ti.i32) -> ti.i32:
        return ti.max(0, ti.min(self.nr - 1, i))

    @ti.func
    def clamp_z_index(self, k: ti.i32) -> ti.i32:
        return ti.max(0, ti.min(self.nz - 1, k))

    @ti.func
    def radial_coordinate(self, i: ti.i32) -> ti.f64:
        return float(i) * self.dr

    @ti.func
    def axial_coordinate(self, k: ti.i32) -> ti.f64:
        return self.z_min_m + float(k) * self.dz

    @ti.func
    def node_volume(self, i: ti.i32) -> ti.f64:
        """Return the axisymmetric nodal control-volume measure.

        The nodal volume is the ring volume associated with a radial node:

        ``V_i = pi * (r_outer^2 - r_inner^2) * dz``

        This is the correct normalization for axisymmetric charge deposition
        because a scalar node in ``(r, z)`` represents a full azimuthal ring in
        3D physical space.
        """

        r_center = self.radial_coordinate(i)
        r_inner = ti.max(0.0, r_center - 0.5 * self.dr)
        r_outer = ti.min(self.r_max_m, r_center + 0.5 * self.dr)
        return math.pi * (r_outer * r_outer - r_inner * r_inner) * self.dz

    @ti.kernel
    def _clear_charge_kernel(self):
        for i, k in self.rho_charge:
            self.rho_charge[i, k] = 0.0

    @ti.kernel
    def _clear_space_charge_solution_kernel(self):
        for i, k in self.phi_sce:
            self.phi_sce[i, k] = 0.0
            self.E_sce_r[i, k] = 0.0
            self.E_sce_z[i, k] = 0.0

    def clear_charge(self) -> None:
        """Reset the axisymmetric space-charge density field."""

        self._require_pic_fields()
        self._clear_charge_kernel()

    def clear_space_charge_solution(self) -> None:
        """Reset the dynamic space-charge potential and field arrays."""

        self._require_pic_fields()
        self._clear_space_charge_solution_kernel()

    def _require_static_fields(self) -> None:
        if not self.has_static_fields:
            raise RuntimeError("This UnifiedGrid2D instance has PIC-only storage.")

    def _require_pic_fields(self) -> None:
        if not self.has_pic_fields:
            raise RuntimeError("This UnifiedGrid2D instance has static-only storage.")

    @ti.kernel
    def _bake_dummy_fields_kernel(self):
        """Bake placeholder static fields using smooth analytic expressions."""

        for i, k in self.E_dc_r:
            r = self.radial_coordinate(i)
            z = self.axial_coordinate(k)
            z_norm = (z - self.z_min_m) / (self.z_max_m - self.z_min_m)
            r_norm = r / self.r_max_m

            # Analytic derivatives of the dummy potential.
            self.E_dc_r[i, k] = 8.0 * 180.0 * (1.0 - z_norm) * r / (self.r_max_m * self.r_max_m) * ti.exp(
                -4.0 * r_norm * r_norm
            )
            self.E_dc_z[i, k] = 180.0 / (self.z_max_m - self.z_min_m) * ti.exp(-4.0 * r_norm * r_norm)
            self.E_rf_r[i, k] = 0.0
            self.E_rf_z[i, k] = 0.0

            # Placeholder axisymmetric gas field.
            self.P_gas[i, k] = 101325.0 * ti.exp(-2.2 * z_norm) * ti.exp(-1.8 * r_norm * r_norm)
            self.T_gas[i, k] = 300.0 + 35.0 * ti.exp(-3.0 * z_norm) - 8.0 * r_norm * r_norm
            self.v_gas_r[i, k] = 18.0 * r_norm * ti.exp(-2.5 * z_norm)
            self.v_gas_z[i, k] = 220.0 * (1.0 - 0.6 * z_norm) * ti.exp(-1.5 * r_norm * r_norm)

    @ti.kernel
    def _bake_dummy_potentials_kernel(self):
        for i, k in self.phi_dc:
            r = self.radial_coordinate(i)
            z = self.axial_coordinate(k)
            z_norm = (z - self.z_min_m) / (self.z_max_m - self.z_min_m)
            r_norm = r / self.r_max_m
            self.phi_dc[i, k] = 180.0 * (1.0 - z_norm) * ti.exp(-4.0 * r_norm * r_norm)
            self.phi_rf[i, k] = 0.0

    @ti.kernel
    def _bake_zero_static_fields_kernel(self, background_pressure_pa: ti.f64, background_temperature_k: ti.f64):
        """Fill the static buffers with a vacuum-like background.

        This mode is useful for sanity checks where we want to isolate the PIC
        space-charge physics:

        - ``phi_dc = 0`` and ``E_dc = 0`` remove externally applied fields,
        - ``P_gas ~= 0`` suppresses collisions through a vanishing neutral density,
        - ``v_gas = 0`` removes any gas-flow-driven drift.
        """

        for i, k in self.E_dc_r:
            self.E_dc_r[i, k] = 0.0
            self.E_dc_z[i, k] = 0.0
            self.E_rf_r[i, k] = 0.0
            self.E_rf_z[i, k] = 0.0
            self.P_gas[i, k] = background_pressure_pa
            self.T_gas[i, k] = background_temperature_k
            self.v_gas_r[i, k] = 0.0
            self.v_gas_z[i, k] = 0.0

    @ti.kernel
    def _bake_zero_potentials_kernel(self):
        for i, k in self.phi_dc:
            self.phi_dc[i, k] = 0.0
            self.phi_rf[i, k] = 0.0

    @ti.kernel
    def _bake_mach_disk_step_fields_kernel(
        self,
        shock_z_m: ti.f64,
        transition_width_m: ti.f64,
        upstream_pressure_pa: ti.f64,
        downstream_pressure_pa: ti.f64,
        upstream_temperature_k: ti.f64,
        downstream_temperature_k: ti.f64,
        upstream_velocity_z_m_per_s: ti.f64,
        downstream_velocity_z_m_per_s: ti.f64,
    ):
        """Create a smoothed shock-like gas step around a prescribed axial location."""

        for i, k in self.E_dc_r:
            z = self.axial_coordinate(k)
            arg = ti.min(60.0, ti.max(-60.0, (z - shock_z_m) / transition_width_m))
            blend = 1.0 / (1.0 + ti.exp(-arg))

            self.E_dc_r[i, k] = 0.0
            self.E_dc_z[i, k] = 0.0
            self.E_rf_r[i, k] = 0.0
            self.E_rf_z[i, k] = 0.0
            self.P_gas[i, k] = upstream_pressure_pa + blend * (downstream_pressure_pa - upstream_pressure_pa)
            self.T_gas[i, k] = upstream_temperature_k + blend * (downstream_temperature_k - upstream_temperature_k)
            self.v_gas_r[i, k] = 0.0
            self.v_gas_z[i, k] = upstream_velocity_z_m_per_s + blend * (
                downstream_velocity_z_m_per_s - upstream_velocity_z_m_per_s
            )

    def bake_dummy_fields(self) -> None:
        """Public entry point for static placeholder-field generation."""

        self._require_static_fields()
        self._bake_dummy_fields_kernel()
        if self.has_potential_fields:
            self._bake_dummy_potentials_kernel()
        if self.has_pic_fields:
            self.clear_charge()
            self.clear_space_charge_solution()

    def bake_zero_static_fields(self, *, background_pressure_pa: float = 0.0, background_temperature_k: float = 300.0) -> None:
        """Public entry point for vacuum / zero-external-field sanity checks."""

        self._require_static_fields()
        self._bake_zero_static_fields_kernel(float(background_pressure_pa), float(background_temperature_k))
        if self.has_potential_fields:
            self._bake_zero_potentials_kernel()
        if self.has_pic_fields:
            self.clear_charge()
            self.clear_space_charge_solution()

    def bake_mach_disk_step_fields(
        self,
        *,
        shock_z_m: float = 5.0e-3,
        transition_width_m: float = 2.0e-4,
        upstream_pressure_pa: float = 5.0,
        downstream_pressure_pa: float = 250.0,
        upstream_temperature_k: float = 300.0,
        downstream_temperature_k: float = 520.0,
        upstream_velocity_z_m_per_s: float = 650.0,
        downstream_velocity_z_m_per_s: float = 80.0,
    ) -> None:
        """Public entry point for the Mach-disk heating sanity check."""

        self._require_static_fields()
        self._bake_mach_disk_step_fields_kernel(
            float(shock_z_m),
            float(max(transition_width_m, 1.0e-9)),
            float(upstream_pressure_pa),
            float(downstream_pressure_pa),
            float(upstream_temperature_k),
            float(downstream_temperature_k),
            float(upstream_velocity_z_m_per_s),
            float(downstream_velocity_z_m_per_s),
        )
        if self.has_potential_fields:
            self._bake_zero_potentials_kernel()
        if self.has_pic_fields:
            self.clear_charge()
            self.clear_space_charge_solution()
