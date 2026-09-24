# Physics, preserved behavior, and validation limits

This distribution is a publication-oriented copy, not a new numerical method.
Original core scheduling, RK4, field gathering, PIC deposition/Poisson,
boundaries, gas, and iict-lite formulas have not been changed for packaging.

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

## Fragmentation caveat inherited from the current src snapshot

The existing production runtime consumes the backend fragmentation probability
on selected explicit collision updates. It does not implement a separate
all-active-ion hazard integration after every microstep. Thus the backend's
Eyring probability unit tests do not establish collision-independent lifetime
integration in the full runtime. Hybrid-Langevin updates are not proof of
internal-energy thermalization. Fragmentation is disabled in the public demos.

Do not apply an archived docs/src_v5.1 patch to this release by assumption.
A change to the full-runtime fragmentation timing needs its own validation.

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
