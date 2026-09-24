# Third-party boundaries

MuCITE was developed by Dr. Yihui Yan. This source distribution contains the
project's simulation, interface, preprocessing, and diagnostic Python modules.

## IonSPA

No IonSPA package, heat-capacity profile collection, pseudo-atom lookup table,
image, manuscript PDF, or vendored IonSPA metadata is distributed here.
The project-owned compatibility bridge remains under src/env/collisions;
it is distinct from the excluded src/env/collisions/ionspa directory.
It dynamically imports the external `ionspa` package only when requested with
`--collision-physics-backend ionspa --ionspa-backend local`.
Users must obtain authorization for their own provider and resources.
The release never searches a parent checkout or external source directory.

The legacy `approximate` bridge is retained for API regression compatibility.
It is not the iict-lite backend and should not be used as an IonSPA-equivalence
claim. Legacy `bundled` requests fail with an explanatory error in this release.

## Scientific attribution

The independent iict-lite backend cites J. S. Prell's IICT paper:
https://doi.org/10.1016/j.ijms.2024.117290.
Equation references are recorded in its source. Model parameters come from
users or the explicitly labelled demonstration configuration. No IonSPA
resource is required by iict-lite. Numerical equality to IonSPA is not claimed.
Source exclusion and citations do not establish the full legal provenance of
every historical project-owned adapter; this preparation is not a legal audit.

## Dependencies and external data

NumPy, SciPy, Taichi, Matplotlib, pandas, Pillow, PyAMG, and optional h5py/pytest
are installed independently. Their distributions provide their license texts.
This repository does not relicense those projects or bundle a Python runtime.

SIMION and Ansys Fluent are external tools. Their executables, raw potential
arrays, CFD exports, experimental results, and proprietary geometry are not
included. Users supply appropriately authorized data through existing file
interfaces. The demonstration uses generated analytic electric fields and gas.
