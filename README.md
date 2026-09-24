# MuCITE

Ion trajectory simulation, developed by **Dr. Yihui Yan** from Gabelica Group.
MuCITE propagates weighted ion packs in three dimensions, using axisymmetric
electric/gas fields, RF modulation, optional electrostatic PIC, and selectable
collision physics. A Cartesian field/mask overlay is also available.

This public release uses the independent `iict-lite` backend for its
demonstrations. **IonSPA source and data are not included.** MuCITE is licensed
under the GNU General Public License version 3; see [LICENSE.md](LICENSE.md).

## User documentation

[Beginner User Guide](docs/User_Guide_en.md) Includes installation, step-by-step GUI operations, saving/loading and transferring between computers, Beam/Field/PIC/IICT parameters, command-line examples, result interpretation, and troubleshooting.

To understand how the program works, please read
[Source Code Responsibilities and API Guide](docs/Source_Guide_en.md) ·

For the complete file-by-file list, see
[Complete Package Layout](docs/Package_Layout.md).

## Install and run

Use Python **3.11** (Taichi 1.7.4). From this repository's root, on Windows:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e ".[test]"
& .\.venv\Scripts\python.exe -m src --config configs/headless_smoke.json
& .\.venv\Scripts\python.exe -m src.render.gui
```

On Linux use `python3.11 -m venv .venv` and `.venv/bin/python` for the same
commands. The native GUI requires Tk; when the Python Tk module is absent,
MuCITE offers a local browser GUI. The release CI exercises Windows CPU runs.
GPU use requires a compatible local driver and is not a CPU smoke-test claim.

An editable installation is recommended for GUI field baking and project data.
The installed console commands are `mucite` and `mucite-gui`. `python -m src`
runs the short default demo; a packaged demo also supports wheel installation.

```powershell
& .\.venv\Scripts\python.exe -m src --help
& .\.venv\Scripts\python.exe -m src --config configs/headless_smoke.json --report --report-dir outputs/smoke
& .\.venv\Scripts\python.exe -m examples.engine_events
# Equivalent installed entry points:
& .\.venv\Scripts\mucite.exe --help
& .\.venv\Scripts\mucite-gui.exe
```

The demo uses 8 weighted ion packs, seed 7, 1 nanosecond duration, a small
shared static/PIC grid, generated analytic fields/gas, and disabled
fragmentation. It is an installation exercise, not a calibrated S-lens result.
The GUI also opens with a short, synthetic-field demonstration. Supply your
field and mask files and turn off **dummy static field** for instrument studies.

## Collision parameters

The demo explicitly references `configs/iict_lite_parameters.example.json`:
classical heat capacity with 72 atoms, constant pseudo-atom mass 30 Da, and
fragmentation `none`. These are illustrative inputs, not universal defaults
or fitted values for a named analyte. Replace them for your ion/gas system.

```powershell
& .\.venv\Scripts\python.exe -m src --config configs/headless_smoke.json `
  --collision-physics-backend iict-lite `
  --iict-parameter-config configs/iict_lite_parameters.example.json `
  --iict-pseudoatom-mass-da 35
```

CLI overrides take precedence; reports record effective values and sources.
Supported heat models are `classical`, `constant_cv`, and `tabulated`.
Pseudo-atom mass may be constant or a rectangular temperature/speed table.
Fragmentation may be `none` or user-parameterized `eyring`. See
[physics and limits](docs/Physics_and_Limits.md) before enabling fragmentation.
No user formula is evaluated with `eval`.

Optional legacy compatibility, using a separately obtained lawful provider:

```powershell
& .\.venv\Scripts\python.exe -m src --config configs/headless_smoke.json `
  --collision-physics-backend ionspa --ionspa-backend local
```

The legacy selector emits a deprecation warning. An absent provider raises an
explicit error. Old `bundled` configurations fail rather than silently becoming
iict-lite. Historical Python dataclass defaults are retained, so direct API
users should load an explicit JSON document as shown in `examples/`.

## Real fields and numerical conventions

Use the Field Baker or `python -m src.data.tools.field_baker --help` to import
your own SIMION potential exports and optional Fluent gas field. No instrument
geometry, PA files, CFD datasets, cached masks, paper PDFs, or experimental
results are distributed. Run commands from the repository root; use absolute
paths for data outside it. Only load trusted project/baked-field files: the
historical NumPy artifact formats can contain pickled objects.

Internal units are SI. SIMION x/y/z maps to Python z/x/y, including velocities.
RF inputs use single-phase **Vpeak**, with
`E = E_dc + E_rf_basis*(Vpeak/Vref)*cos(2*pi*f*t + phase) + E_sc`.
RF is reevaluated at each RK4 stage; PIC is refreshed per macro step.
Static/PIC grids share by default; positive node-count pairs explicitly request
a mesh. Keep coordinate origin, RF polarity, phase, and field metadata consistent
with your input files. The historical +90/-90 degree SIMION comparison was an
empirical convention for a specific private dataset, not a universal mapping.

## Development and validation

```powershell
& .\.venv\Scripts\python.exe -m compileall -q src tests examples tools
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe tools/check_release.py
& .\.venv\Scripts\python.exe tools/gui_smoke.py
```

The included tests cover independent IICT models and numerical checks,
synthetic fields/transport, PIC, configuration, GUI logic, and distribution
boundaries. They do not require IonSPA or the private final SIMION benchmark.
See [physics and limitations](docs/Physics_and_Limits.md) and the
[package layout](docs/Package_Layout.md).

Local release checks: **281 tests passed**; the fixed-seed CPU smoke completed
107 collisions with 8 surviving packs. The installed wheel reproduced those
counts. The installed `mucite` command, GUI construction, source exclusions,
license metadata, and package builds passed.
Generate a clean repository ZIP with `python tools/package_github.py`.

## Attribution

Please cite this software using `CITATION.cff` and the IICT paper by J. S. Prell,
DOI [10.1016/j.ijms.2024.117290](https://doi.org/10.1016/j.ijms.2024.117290).
`iict-lite` is a paper-driven independent approximation; it does not claim
numerical equivalence to IonSPA. See [third-party notices](THIRD_PARTY_NOTICES.md).
MuCITE-owned code is distributed under the
[GNU General Public License version 3](LICENSE.md); external programs, optional
providers, and user-supplied data remain subject to their own terms.
