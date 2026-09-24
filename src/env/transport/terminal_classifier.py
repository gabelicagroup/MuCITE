"""Taichi endpoint terminal-event classification."""

import taichi as ti


class TerminalClassifierMixin:
    """Electrode, detector, radial, and domain endpoint classification."""

    @ti.func
    def _classify_electrode3d(
        self,
        x: ti.f64,
        y: ti.f64,
        z: ti.f64,
        hit_distance_m: ti.f64,
        active_code: ti.i32,
        electrode_code: ti.i32,
    ) -> ti.types.vector(3, ti.f64):
        status, electrode_id, distance_m = active_code, -1, 1.0e99
        if ti.static(self.has_electrode_mask3d):
            inside = 0
            if x >= self.mask3d_x_min_m:
                if x <= self.mask3d_x_max_m:
                    if y >= self.mask3d_y_min_m:
                        if y <= self.mask3d_y_max_m:
                            if z >= self.mask3d_z_min_m:
                                if z <= self.mask3d_z_max_m:
                                    inside = 1
            if inside != 0:
                fx = (x - self.mask3d_x_min_m) / self.mask3d_dx_m
                fy = (y - self.mask3d_y_min_m) / self.mask3d_dy_m
                fz = (z - self.mask3d_z_min_m) / self.mask3d_dz_m
                ix = ti.max(0, ti.min(self.mask3d_nx - 1, int(ti.floor(fx + 0.5))))
                iy = ti.max(0, ti.min(self.mask3d_ny - 1, int(ti.floor(fy + 0.5))))
                iz = ti.max(0, ti.min(self.mask3d_nz - 1, int(ti.floor(fz + 0.5))))
                electrode_id = self.electrode3d_id[ix, iy, iz]
                distance_m = self._scalar_trilinear_electrode3d(
                    self.electrode3d_surface_distance, x, y, z,
                )
                if self.electrode3d_metal_mask[ix, iy, iz] != 0 or distance_m <= hit_distance_m:
                    status = electrode_code
        return ti.Vector([float(status), float(electrode_id), distance_m])

    @ti.func
    def _classify_electrode2d(
        self,
        r: ti.f64,
        z: ti.f64,
        hit_distance_m: ti.f64,
        use_mask: ti.i32,
        active_code: ti.i32,
        electrode_code: ti.i32,
        current: ti.template(),
    ) -> ti.types.vector(3, ti.f64):
        status, electrode_id, distance_m = int(current[0]), int(current[1]), current[2]
        if status == active_code and use_mask != 0:
            inside = 0
            if r >= self.mask_r_min_m:
                if r <= self.mask_r_max_m:
                    if z >= self.mask_z_min_m:
                        if z <= self.mask_z_max_m:
                            inside = 1
            if inside != 0:
                sample = self._sample_electrode2d(r, z)
                electrode_id, distance_m = int(sample[0]), sample[1]
                if sample[2] != 0.0 or distance_m <= hit_distance_m:
                    status = electrode_code
        return ti.Vector([float(status), float(electrode_id), distance_m])

    @ti.func
    def _sample_electrode2d(
        self,
        r: ti.f64,
        z: ti.f64,
    ) -> ti.types.vector(3, ti.f64):
        fr = (r - self.mask_r_min_m) / self.mask_dr_m
        fz = (z - self.mask_z_min_m) / self.mask_dz_m
        ir_hi = ti.max(1, ti.min(self.mask_nr - 1, int(ti.ceil(fr))))
        iz_hi = ti.max(1, ti.min(self.mask_nz - 1, int(ti.ceil(fz))))
        ir_lo, iz_lo = ir_hi - 1, iz_hi - 1
        r_hi = self.mask_r_min_m + float(ir_hi) * self.mask_dr_m
        z_hi = self.mask_z_min_m + float(iz_hi) * self.mask_dz_m
        r_lo = self.mask_r_min_m + float(ir_lo) * self.mask_dr_m
        z_lo = self.mask_z_min_m + float(iz_lo) * self.mask_dz_m
        ir_nearest, iz_nearest = ir_lo, iz_lo
        if ti.abs(r_hi - r) < ti.abs(r - r_lo):
            ir_nearest = ir_hi
        if ti.abs(z_hi - z) < ti.abs(z - z_lo):
            iz_nearest = iz_hi
        i0 = ti.max(0, ti.min(self.mask_nr - 2, int(ti.floor(fr))))
        k0 = ti.max(0, ti.min(self.mask_nz - 2, int(ti.floor(fz))))
        i1, k1 = i0 + 1, k0 + 1
        wr, wz = fr - float(i0), fz - float(k0)
        distance_m = (
            (1.0 - wr) * (1.0 - wz) * self.electrode_surface_distance[i0, k0]
            + wr * (1.0 - wz) * self.electrode_surface_distance[i1, k0]
            + (1.0 - wr) * wz * self.electrode_surface_distance[i0, k1]
            + wr * wz * self.electrode_surface_distance[i1, k1]
        )
        metal = self.electrode_metal_mask[ir_nearest, iz_nearest]
        return ti.Vector([float(self.electrode_id[ir_nearest, iz_nearest]), distance_m, float(metal)])

    @ti.func
    def _classify_domain(
        self,
        status: ti.i32,
        r: ti.f64,
        z: ti.f64,
        detector_z_m: ti.f64,
        detector_radius_m: ti.f64,
        radial_limit_m: ti.f64,
        domain_radius_m: ti.f64,
        domain_length_m: ti.f64,
        active_code: ti.i32,
        z_exit_code: ti.i32,
        radial_out_code: ti.i32,
        domain_out_code: ti.i32,
    ) -> ti.i32:
        result = status
        if result == active_code and (r > radial_limit_m or (z >= detector_z_m and r > detector_radius_m)):
            result = radial_out_code
        elif result == active_code and z >= detector_z_m:
            result = z_exit_code
        elif result == active_code and (z < 0.0 or z > domain_length_m):
            result = domain_out_code
        elif result == active_code and r > domain_radius_m:
            result = radial_out_code
        return result

    @ti.kernel
    def select_terminal_boundary_hits(
        self,
        ions: ti.template(),
        detector_z_m: ti.f64,
        detector_radius_m: ti.f64,
        radial_limit_m: ti.f64,
        domain_radius_m: ti.f64,
        domain_length_m: ti.f64,
        electrode_hit_distance_m: ti.f64,
        use_electrode_mask: ti.i32,
        particle_active_code: ti.i32,
        particle_z_exit_code: ti.i32,
        particle_radial_out_code: ti.i32,
        particle_electrode_hit_code: ti.i32,
        particle_domain_out_code: ti.i32,
    ):
        ions.terminal_count[None] = 0
        for p in range(ions.n_particles):
            if ions.active[p] != 0:
                x, y, z = ions.x[p], ions.y[p], ions.z[p]
                r = ti.sqrt(x * x + y * y)
                result = self._classify_electrode3d(
                    x, y, z, electrode_hit_distance_m,
                    particle_active_code, particle_electrode_hit_code,
                )
                result = self._classify_electrode2d(
                    r, z, electrode_hit_distance_m, use_electrode_mask,
                    particle_active_code, particle_electrode_hit_code, result,
                )
                status = self._classify_domain(
                    int(result[0]), r, z, detector_z_m, detector_radius_m,
                    radial_limit_m, domain_radius_m, domain_length_m,
                    particle_active_code, particle_z_exit_code,
                    particle_radial_out_code, particle_domain_out_code,
                )
                if status != particle_active_code:
                    ions.active[p] = 0
                    j = ti.atomic_add(ions.terminal_count[None], 1)
                    ions.terminal_indices[j] = p
                    ions.terminal_status[j] = status
                    ions.terminal_electrode_id[j] = int(result[1])
                    ions.terminal_surface_distance[j] = result[2]
