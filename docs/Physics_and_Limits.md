# Physics, preserved behavior, and validation limits

MuCITE 2.0 keeps the RK4 field pusher, axisymmetric PIC and field coordinate
conventions. Collision-rate estimation, hybrid updates and dissociation
scheduling differ from the earlier public release.

## Independent IICT

The implementation cites Prell (2024), DOI 10.1016/j.ijms.2024.117290.
`src/env/collisions/iict_lite/physics.py` labels collision/energy equations;
the backend shares the numerical core between scalar and batch APIs.
Gas bulk velocity remains three-dimensional, with gas and pseudo-atom thermal
velocities sampled separately. The existing Poisson scheduler selects events.
The backend operates on an already selected collision.

Heat capacity, pseudo-atom mass, and Eyring activation parameters are user
inputs. The included 72-atom/30-Da example is only a runnable illustration.
Classical heat capacity is `(3*num_atoms-6)*kB` per ion. Tables must have explicit
SI/Da units and satisfy the implementation's finite-value, monotonicity, grid,
and range checks.

## Collision rates, hybrid updates, and dissociation

The collision rate is number density times cross section times the mean
relative speed to the neutral Maxwell distribution. At zero drift it is
nonzero at positive gas temperature and density. The time-step constraint
and explicit event sampler share this model.

The experimental hybrid operator uses joint velocity and internal-energy
moments from the IICT collision rule. It uses Gaussian steps only where the
expected event count, momentum relaxation, and energy guards permit; otherwise
it resolves binary events. A hybrid window does not guarantee any Gaussian
steps, speedup, or agreement with an explicit run.

Dissociation is integrated separately from collision occurrence. The hazard
increases by the rate at post-collision internal temperature times the time
alive in the step. Terminal times truncate the increment; birth is not
retroactive. Parent weight decays exponentially. Transport mode retains
total transport weight and PIC charge, loss mode removes fragmented weight,
and off bypasses dissociation. No separate fragment trajectories are spawned.
The public demos keep fragmentation disabled.

The splitting is still collision-before-RK4. Validate the collision and
dissociation results under time-step refinement for the regime being studied;
the new integration does not make Eyring parameters universal.

## Time and mesh error

RF is sampled at RK4 stage times; the PIC self field is frozen over a macro
step. RK4 is non-symplectic, and this field refresh scheme is not a guarantee
of total energy conservation. Converge both micro
step and PIC macro step, as well as mesh and weighted-ion-pack population.
RF forcing, gas collisions, open boundaries, and source injection physically
change particle energy; raw kinetic-energy drift is not by itself an error
measure in the coupled model.

## Historical baseline

The development repository reports a private pure-electric SIMION comparison:
989 transmitted and 11 electrode hits for both codes, with matching hit IDs;
TOF p95 absolute difference 0.00210 us and KE p95 difference 0.00892 eV.
Its input rays, baked fields, and geometry are not part of this distribution.


The runnable release tests use synthetic inputs. Author-supplied experimental
data would require a separate, intentional distribution and data-license review.
