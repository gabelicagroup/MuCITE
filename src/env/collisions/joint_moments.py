"""Conditional velocity/internal-energy jump moments of the existing IICT rule.

Neutral and pseudo-atom velocities have exactly the distributions used by the
binary backends. This is a diffusion approximation to that rule, not a new
cross section, heat capacity, or activation model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import Boltzmann


@dataclass(frozen=True)
class IictJumpMoments:
    """Per-collision conditional moments; energy is J/ion throughout."""

    momentum_fraction: np.ndarray
    mean_velocity: np.ndarray
    mean_energy_j: np.ndarray
    thermal_velocity_variance: np.ndarray
    raw_velocity_energy: np.ndarray
    raw_energy_second_j2: np.ndarray

    def take(self, offsets: np.ndarray) -> "IictJumpMoments":
        """Select rows without recomputing local thermodynamic coefficients."""
        return IictJumpMoments(*(getattr(self, name)[offsets] for name in (
            "momentum_fraction", "mean_velocity", "mean_energy_j",
            "thermal_velocity_variance", "raw_velocity_energy", "raw_energy_second_j2")))

    def raw_second_moment(self) -> np.ndarray:
        """Return E[dZ dZ^T], not Cov(dZ), for Z=(vx, vy, vz, U)."""
        count = self.mean_energy_j.size
        result = np.zeros((count, 4, 4))
        result[:, :3, :3] = np.einsum("ni,nj->nij", self.mean_velocity, self.mean_velocity)
        result[:, :3, :3] += self.thermal_velocity_variance[:, None, None] * np.eye(3)
        result[:, :3, 3] = result[:, 3, :3] = self.raw_velocity_energy
        result[:, 3, 3] = self.raw_energy_second_j2
        return result


def iict_jump_moments(
    relative_velocity_m_per_s: np.ndarray, ion_mass_kg: np.ndarray,
    internal_temperature_k: np.ndarray, gas_temperature_k: np.ndarray,
    gas_mass_kg: float, pseudoatom_mass_kg: np.ndarray,
) -> IictJumpMoments:
    """Analytic Gaussian linear/quadratic moments of one existing IICT impulse."""
    w = np.asarray(relative_velocity_m_per_s, dtype=np.float64)
    mass, ti, tg, ma = np.broadcast_arrays(
        ion_mass_kg, internal_temperature_k, gas_temperature_k, pseudoatom_mass_kg)
    if (w.shape != (mass.size, 3) or not np.all(np.isfinite(w))
            or any(not np.all(np.isfinite(x)) or np.any(x <= 0) for x in (mass, ti, tg, ma))
            or not np.isfinite(gas_mass_kg) or gas_mass_kg <= 0 or np.any(ma > mass)):
        raise ValueError("IICT joint moments require finite positive masses/temperatures and 3D velocities.")
    b = 2 * gas_mass_kg / (ma + gas_mass_kg)
    f = ma / mass
    c = f * b
    atom_var = Boltzmann * ti / ma
    gas_var = Boltzmann * tg / gas_mass_kg
    h_var = (1 - b)**2 * atom_var + b**2 * gas_var
    w2 = np.einsum("ni,ni->n", w, w)
    mu_v = -c[:, None] * w
    mu_u = .5 * ma * (3 * (h_var - atom_var) + b**2 * w2)
    # Stable non-negative expression for the centered quadratic energy variance.
    quadratic_var = 1.5 * ma**2 * ((h_var - atom_var)**2 + 2 * atom_var * b**2 * gas_var)
    var_u = quadratic_var + ma**2 * b**2 * h_var * w2
    cross = mu_v * mu_u[:, None] - (f * ma * b * h_var)[:, None] * w
    return IictJumpMoments(c, mu_v, mu_u, f**2 * h_var, cross, var_u + mu_u**2)


def _decay_average(value: np.ndarray) -> np.ndarray:
    """Return (1-exp(-x))/x without cancellation at x=0."""
    return np.divide(-np.expm1(-value), value, out=np.ones_like(value), where=value != 0)


def gaussian_joint_step(
    relative_velocity_m_per_s: np.ndarray, energy_j: np.ndarray,
    moments: IictJumpMoments, expected_collisions: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a locally frozen joint diffusion, with exponential v drift.

    The diffusion uses raw jump moments, including random collision-count
    fluctuations. Cross noise couples v and U. Only the frozen linear SDE is
    integrated exactly; state-dependent coefficients still require convergence.
    """
    w = np.asarray(relative_velocity_m_per_s)
    lam = np.asarray(expected_collisions, dtype=np.float64)
    c = moments.momentum_fraction
    relax = c * lam
    vv_scale = lam * _decay_average(2 * relax)
    cross = (lam * _decay_average(relax))[:, None] * moments.raw_velocity_energy
    isotropic = vv_scale * moments.thermal_velocity_variance
    w2 = np.einsum("ni,ni->n", w, w)
    parallel = isotropic + vv_scale * c**2 * w2
    uu = lam * moments.raw_energy_second_j2
    conditional = uu - np.divide(np.einsum("ni,ni->n", cross, cross), parallel,
                                 out=np.zeros_like(uu), where=parallel > 0)
    if np.any(conditional < -1e-12 * np.maximum(uu, np.finfo(float).tiny)):
        raise FloatingPointError("Joint collision covariance is not positive semidefinite.")
    z = rng.normal(size=(w.shape[0], 5))
    v_noise = np.sqrt(isotropic)[:, None] * z[:, :3]
    v_noise += (np.sqrt(vv_scale) * c * z[:, 3])[:, None] * w
    u_noise = np.divide(np.einsum("ni,ni->n", cross, v_noise), parallel,
                        out=np.zeros_like(uu), where=parallel > 0)
    u_noise += np.sqrt(np.maximum(conditional, 0)) * z[:, 4]
    new_u = energy_j + lam * moments.mean_energy_j + u_noise
    if np.any(~np.isfinite(new_u)) or np.any(new_u <= 0):
        raise FloatingPointError("Joint Gaussian energy left its admissible domain; tighten hybrid controls.")
    return np.exp(-relax)[:, None] * w + v_noise, new_u


def gaussian_collision_window(
    energy_j: np.ndarray, moments: IictJumpMoments, *,
    max_relaxation: float, max_energy_fraction: float,
    energy_bounds_j: tuple[float, float],
) -> np.ndarray:
    """Limit expected counts before drawing, with an eight-sigma table guard.

    No rejected noise is redrawn and no energy is clipped. If this admissible
    window is too short for diffusion, the runtime uses actual binary events.
    """
    mu = np.abs(moments.mean_energy_j)
    second = moments.raw_energy_second_j2
    margin = np.maximum(np.minimum(energy_j - energy_bounds_j[0],
                                  energy_bounds_j[1] - energy_j), 0)
    drift_budget = np.minimum(max_energy_fraction * energy_j, margin / 2)
    noise_budget = np.minimum(max_energy_fraction * energy_j, margin / 16)
    drift_limit = np.divide(drift_budget, mu, out=np.full_like(mu, np.inf), where=mu > 0)
    noise_limit = np.divide(noise_budget**2, second, out=np.full_like(mu, np.inf), where=second > 0)
    return np.minimum(max_relaxation / moments.momentum_fraction,
                      np.minimum(drift_limit, noise_limit))
