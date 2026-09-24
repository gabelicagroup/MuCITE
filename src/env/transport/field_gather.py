"""Taichi field interpolation and axisymmetric-to-Cartesian mapping."""

import taichi as ti


class FieldGatherMixin:
    """Field interpolation behavior shared by transport kernels."""

    @ti.func
    def _scalar_trilinear_cartesian3d(
        self,
        field: ti.template(),
        x_m: ti.f64,
        y_m: ti.f64,
        z_m: ti.f64,
    ) -> ti.f64:
        x = ti.max(self.field3d_x_min_m, ti.min(self.field3d_x_max_m, x_m))
        y = ti.max(self.field3d_y_min_m, ti.min(self.field3d_y_max_m, y_m))
        z = ti.max(self.field3d_z_min_m, ti.min(self.field3d_z_max_m, z_m))
        fx = (x - self.field3d_x_min_m) / self.field3d_dx_m
        fy = (y - self.field3d_y_min_m) / self.field3d_dy_m
        fz = (z - self.field3d_z_min_m) / self.field3d_dz_m
        i0 = ti.max(0, ti.min(self.field3d_nx - 2, int(ti.floor(fx))))
        j0 = ti.max(0, ti.min(self.field3d_ny - 2, int(ti.floor(fy))))
        k0 = ti.max(0, ti.min(self.field3d_nz - 2, int(ti.floor(fz))))
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        wx, wy, wz = fx - float(i0), fy - float(j0), fz - float(k0)
        c00 = (1.0 - wx) * field[i0, j0, k0] + wx * field[i1, j0, k0]
        c10 = (1.0 - wx) * field[i0, j1, k0] + wx * field[i1, j1, k0]
        c01 = (1.0 - wx) * field[i0, j0, k1] + wx * field[i1, j0, k1]
        c11 = (1.0 - wx) * field[i0, j1, k1] + wx * field[i1, j1, k1]
        c0 = (1.0 - wy) * c00 + wy * c10
        c1 = (1.0 - wy) * c01 + wy * c11
        return (1.0 - wz) * c0 + wz * c1

    @ti.func
    def _scalar_trilinear_electrode3d(
        self,
        field: ti.template(),
        x_m: ti.f64,
        y_m: ti.f64,
        z_m: ti.f64,
    ) -> ti.f64:
        x = ti.max(self.mask3d_x_min_m, ti.min(self.mask3d_x_max_m, x_m))
        y = ti.max(self.mask3d_y_min_m, ti.min(self.mask3d_y_max_m, y_m))
        z = ti.max(self.mask3d_z_min_m, ti.min(self.mask3d_z_max_m, z_m))
        fx = (x - self.mask3d_x_min_m) / self.mask3d_dx_m
        fy = (y - self.mask3d_y_min_m) / self.mask3d_dy_m
        fz = (z - self.mask3d_z_min_m) / self.mask3d_dz_m
        i0 = ti.max(0, ti.min(self.mask3d_nx - 2, int(ti.floor(fx))))
        j0 = ti.max(0, ti.min(self.mask3d_ny - 2, int(ti.floor(fy))))
        k0 = ti.max(0, ti.min(self.mask3d_nz - 2, int(ti.floor(fz))))
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        wx, wy, wz = fx - float(i0), fy - float(j0), fz - float(k0)
        c00 = (1.0 - wx) * field[i0, j0, k0] + wx * field[i1, j0, k0]
        c10 = (1.0 - wx) * field[i0, j1, k0] + wx * field[i1, j1, k0]
        c01 = (1.0 - wx) * field[i0, j0, k1] + wx * field[i1, j0, k1]
        c11 = (1.0 - wx) * field[i0, j1, k1] + wx * field[i1, j1, k1]
        c0 = (1.0 - wy) * c00 + wy * c10
        c1 = (1.0 - wy) * c01 + wy * c11
        return (1.0 - wz) * c0 + wz * c1

    @ti.func
    def _scalar_bilinear_static(
        self,
        field: ti.template(),
        r_m: ti.f64,
        z_m: ti.f64,
    ) -> ti.f64:
        r = ti.max(0.0, ti.min(self.static_grid.r_max_m, r_m))
        z = ti.max(self.static_grid.z_min_m, ti.min(self.static_grid.z_max_m, z_m))
        fr = r / self.static_grid.dr
        fz = (z - self.static_grid.z_min_m) / self.static_grid.dz
        i0 = ti.max(0, ti.min(self.static_grid.nr - 2, int(ti.floor(fr))))
        k0 = ti.max(0, ti.min(self.static_grid.nz - 2, int(ti.floor(fz))))
        i1, k1 = i0 + 1, k0 + 1
        wr, wz = fr - float(i0), fz - float(k0)
        return (
            (1.0 - wr) * (1.0 - wz) * field[i0, k0]
            + wr * (1.0 - wz) * field[i1, k0]
            + (1.0 - wr) * wz * field[i0, k1]
            + wr * wz * field[i1, k1]
        )

    @ti.func
    def _scalar_bilinear_pic(
        self,
        field: ti.template(),
        r_m: ti.f64,
        z_m: ti.f64,
    ) -> ti.f64:
        r = ti.max(0.0, ti.min(self.pic_grid.r_max_m, r_m))
        z = ti.max(self.pic_grid.z_min_m, ti.min(self.pic_grid.z_max_m, z_m))
        fr = r / self.pic_grid.dr
        fz = (z - self.pic_grid.z_min_m) / self.pic_grid.dz
        i0 = ti.max(0, ti.min(self.pic_grid.nr - 2, int(ti.floor(fr))))
        k0 = ti.max(0, ti.min(self.pic_grid.nz - 2, int(ti.floor(fz))))
        i1, k1 = i0 + 1, k0 + 1
        wr, wz = fr - float(i0), fz - float(k0)
        return (
            (1.0 - wr) * (1.0 - wz) * field[i0, k0]
            + wr * (1.0 - wz) * field[i1, k0]
            + (1.0 - wr) * wz * field[i0, k1]
            + wr * wz * field[i1, k1]
        )

    @ti.func
    def _axisymmetric_fields(
        self,
        ion_x: ti.f64,
        ion_y: ti.f64,
        ion_z: ti.f64,
        r: ti.f64,
        rf_modulation: ti.f64,
    ) -> ti.types.vector(8, ti.f64):
        er = (
            self._scalar_bilinear_static(self.static_grid.E_dc_r, r, ion_z)
            + rf_modulation * self._scalar_bilinear_static(self.static_grid.E_rf_r, r, ion_z)
            + self._scalar_bilinear_pic(self.pic_grid.E_sce_r, r, ion_z)
        )
        ez = (
            self._scalar_bilinear_static(self.static_grid.E_dc_z, r, ion_z)
            + rf_modulation * self._scalar_bilinear_static(self.static_grid.E_rf_z, r, ion_z)
            + self._scalar_bilinear_pic(self.pic_grid.E_sce_z, r, ion_z)
        )
        vgr = self._scalar_bilinear_static(self.static_grid.v_gas_r, r, ion_z)
        vgz = self._scalar_bilinear_static(self.static_grid.v_gas_z, r, ion_z)
        temperature = self._scalar_bilinear_static(self.static_grid.T_gas, r, ion_z)
        pressure = self._scalar_bilinear_static(self.static_grid.P_gas, r, ion_z)
        ex, ey, vgx, vgy = 0.0, 0.0, 0.0, 0.0
        if r > 1.0e-16:
            ex, ey = er * ion_x / r, er * ion_y / r
            vgx, vgy = vgr * ion_x / r, vgr * ion_y / r
        return ti.Vector([ex, ey, ez, vgx, vgy, vgz, temperature, pressure])

    @ti.func
    def _add_cartesian_overlay(
        self,
        fields: ti.template(),
        x: ti.f64,
        y: ti.f64,
        z: ti.f64,
        rf_modulation: ti.f64,
    ) -> ti.types.vector(8, ti.f64):
        fields[0] += self._scalar_trilinear_cartesian3d(self.E3d_dc_x, x, y, z)
        fields[1] += self._scalar_trilinear_cartesian3d(self.E3d_dc_y, x, y, z)
        fields[2] += self._scalar_trilinear_cartesian3d(self.E3d_dc_z, x, y, z)
        fields[0] += rf_modulation * self._scalar_trilinear_cartesian3d(self.E3d_rf_x, x, y, z)
        fields[1] += rf_modulation * self._scalar_trilinear_cartesian3d(self.E3d_rf_y, x, y, z)
        fields[2] += rf_modulation * self._scalar_trilinear_cartesian3d(self.E3d_rf_z, x, y, z)
        return fields

    @ti.func
    def gather_fields(
        self,
        ion_x: ti.f64,
        ion_y: ti.f64,
        ion_z: ti.f64,
        time_s: ti.f64,
        rf_angular_frequency_rad_s: ti.f64,
        rf_phase_rad: ti.f64,
        rf_peak_to_reference_scale: ti.f64,
    ) -> ti.types.vector(8, ti.f64):
        """Return ``[Ex,Ey,Ez,vgx,vgy,vgz,T,P]`` at one particle."""

        r = ti.sqrt(ion_x * ion_x + ion_y * ion_y)
        modulation = rf_peak_to_reference_scale * ti.cos(
            rf_angular_frequency_rad_s * time_s + rf_phase_rad
        )
        fields = self._axisymmetric_fields(
            ion_x,
            ion_y,
            ion_z,
            r,
            modulation,
        )
        if ti.static(self.has_cartesian_field3d):
            fields = self._add_cartesian_overlay(
                fields,
                ion_x,
                ion_y,
                ion_z,
                modulation,
            )
        return fields
