# MuCITE 2.0.0

MuCITE 2.0.0 brings the current research runtime's collision, dissociation,
and transfer improvements into the standalone GPLv3 public distribution.
The original [user guide](docs/User_Guide_en.md) and installation workflow
are retained. This document supersedes earlier descriptions of collision
timing and fragmentation.

## Simulation

- **Physical-time dissociation.** Every active transport interval contributes
  `dissociation_hazard += k(T_int) * dt_alive`, even without a collision.
  A terminating pack contributes only until its terminal time; capillary
  buffer time and time before injection are not counted. In transport mode,
  the parent survival fraction is `exp(-dissociation_hazard)` while total
  transport weight and PIC charge are unchanged. Loss mode removes fragmented
  weight; neither mode creates separate fragment particles.
- **Thermal relative-speed collision rates.** The mean relative speed
  integrates neutral Maxwell velocities instead of using only bulk drift.
  The CPU and Taichi time-step calculations and collision selection use the
  same rate, including the zero-drift thermal limit.
- **Experimental joint hybrid operator.** In the hybrid window, the model
  updates both velocity and internal energy using IICT jump moments and their
  cross covariance. When aggregation fails the guards, it uses explicit binary
  events. This replaces the earlier velocity-only thermostat. It does not
  guarantee acceleration or agreement with experiment.
- **Lower data-transfer overhead.** Active-slot and terminal-event transfers
  are compact; empty terminal batches skip transfers. Microstep caches and
  exact-temperature dissociation-rate caching avoid redundant work without
  skipping elapsed hazard time. A global microstep is still used.
- **Diagnostics.** Add gas/electric field and ion-snapshot overlays, weighted
  density comparisons with gas streamlines, field/mask alignment and Fluent
  Mach plots, and spatial precursor-survival/internal-temperature maps. The
  tools accept user-supplied fields and case directories, not bundled data.
  Run `python -m src.data.diagnostics.plot_spatial_survival_temperature --help`
  for one example.

## New Controls

| CLI flag | JSON simulation key | Default |
| --- | --- | ---: |
| `--dissociation-check-every` | `dissociation_check_every` | 100 |
| `--langevin-min-collisions` | `langevin_min_collisions` | 8 |
| `--langevin-max-relaxation` | `langevin_max_relaxation` | 0.1 |
| `--langevin-max-energy-fraction` | `langevin_max_energy_fraction` | 0.05 |

The first control sets the cadence of diagnostic survival checks, not hazard
updates. `0` keeps final/terminal checks; `1` checks every microstep.
The remaining controls bound Gaussian aggregation, not the underlying binary
collision physics. They supplement the existing hybrid window and time-step
controls. Use the CLI or the headless JSON to override these values.

## Upgrading

- Install using Python 3.11; `mucite`, `mucite-gui`, and `python -m src`
  remain available. The packaged demonstration uses independent `iict-lite`
  and synthetic inputs. No IonSPA code or data, private fields/geometry,
  manuscripts, experiment results, or Python runtime are shipped.
- The installed CLI now returns exit code 0 on successful runs instead of
  printing the returned simulation object as an error. Numerical failures
  return a nonzero status; the Python `run_cli()` API still returns its result.
- Use named columns for terminal events and trajectories, and snapshot column
  names from `metadata.json`. The added `dissociation_hazard` field is
  dimensionless; the NPY snapshot schema is version 3. Keep old raw outputs
  intact and write comparisons to new directories.
- Do not expect seed-for-seed agreement with 0.1.0, even with identical
  settings. The rate, event scheduling, and dissociation semantics changed.
  Compare at matched physical times and check time-step and population
  convergence before using results quantitatively.
- An authorized local IonSPA provider is optional. The updated bridge checks
  tabulated temperature/energy ranges rather than clipping out-of-range
  states. Removed bundled-provider requests are still explicitly rejected.
  Public checks use the independent backend or test doubles; an external
  provider is not covered by the distribution's smoke test.

The release retains the previous coordinate, Vpeak, and PIC mesh conventions.
See [physics and limits](docs/Physics_and_Limits.md) for validation scope.

## Verification

The standalone source suite passed 424 tests on Windows, Python 3.11.9,
and Taichi 1.7.4. Tk workbench and dialog construction passed. The public
synthetic CPU example reached 1 ns with 8 survivors and 107 collisions,
without importing or opening IonSPA resources. This is not a private
instrument benchmark or a general performance claim.
