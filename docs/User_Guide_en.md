# MuCITE User Guide: Start Here

Applies to the GitHub release candidate in this repository. Documentation checked on **2026-09-14**. Author: Dr. Yihui Yan.

This guide assumes no prior Python or numerical-simulation experience. Sections 1–5 take you through installation, a small test, the GUI, and output files. Read Sections 6–13 before using real instrument data. To understand or modify the program, continue with the [Source Code Guide](Source_Guide_en.md).

The **repository root** is the directory containing `src`, `configs`, and `pyproject.toml`. After extracting the archive it will normally be named `MuCITE`; inside the author's development project it is named `github_release`. It is not the parent development directory and not the `src` directory.

## Contents

1. What MuCITE does
2. First-time installation
3. First run without field files
4. Using the GUI
5. Saving, loading, and moving to another computer
6. Units and essential terminology
7. Beam and ion source
8. Field Baker, gas, and electrode boundaries
9. Runtime, PIC, and RF
10. Collisions, internal energy, and fragmentation
11. Command line and configuration files
12. Reading results, snapshots, and current
13. Moving from the demo to research
14. Troubleshooting
15. One-page checklist

## 1. What MuCITE does

MuCITE simulates charged-ion motion under electric fields, gas interactions, and the electric field of other ions. It is intended primarily for ESI-MS interfaces and S-lens transport studies.

Think of it as a virtual experiment:

1. Define the ion and its source.
2. Define the electric field, gas, and boundaries.
3. Advance time in small steps, updating position, velocity, collisions, and internal temperature.
4. Record which ions reach the exit, strike electrodes, or remain active.
5. Write plots, tables, snapshots, and reports.

Particles move in three-dimensional `(x,y,z)` space. Common electric and gas fields are stored on a two-dimensional axisymmetric `(r,z)` grid, where `r = sqrt(x²+y²)`. A 2D field does not restrict particles to a plane. Optional Cartesian 3D field and mask interfaces also exist, but they are not part of the beginner workflow.

The public release does **not** contain your SIMION PA/patxt files, Fluent CFD data, instrument geometry, large development fields, manuscript results, IonSPA source/data, Python runtime, or GPU drivers. The first run uses independent `iict-lite` collision physics and generated demonstration fields. It does not require IonSPA.

`iict-lite` is an independent, paper-driven approximation and is not claimed to be numerically equivalent to IonSPA. A successful demo proves that the installed software path works; it does not calibrate your instrument or analyte. See [Physics and Limits](Physics_and_Limits.md).

## 2. First-time installation

### 2.1 Prepare the directory

Extract the complete release archive, for example to `D:\MuCITE`. Do not run it from an archive-preview window. Confirm that the directory contains `README.md`, `pyproject.toml`, `src`, and `configs`. Do not copy only `src`.

| Item | Purpose | Should a beginner edit it? |
| --- | --- | --- |
| `src/` | Application source | No, for normal use |
| `configs/` | Runnable configuration and IICT examples | Edit copies |
| `examples/` | Small Python API examples | Read later |
| `docs/` | User, physics, and source documentation | Read |
| `tests/` | Software checks | Do not edit |
| `tools/` | Release, inventory, and GUI checks | Development/release only |
| `pyproject.toml` | Package, Python, dependencies, entry points | Do not delete |
| `requirements*.txt`, `requirements.lock` | Dependency records | Do not remove as cleanup |
| `.venv/` | Your local isolated Python environment | Recreate on each computer |
| `outputs/` | Generated results | Back up important runs |

### 2.2 Install Python 3.11

This release requires **64-bit Python 3.11**. Do not replace it with Python 3.12/3.13 merely because they are newer; the package metadata and Taichi compatibility constrain the version.

Open PowerShell in the extracted directory and run:

```powershell
py -3.11 --version
```

You should see `Python 3.11.x`. Otherwise install 64-bit Python 3.11 from the official Python distribution, including Tcl/Tk, and reopen PowerShell.

The `&` below is PowerShell's execution operator and must be included. If the prompt is `>>>`, you are inside Python; enter `exit()` to return to PowerShell.

### 2.3 Install on a new Windows computer

```powershell
Set-Location "D:\MuCITE"
Get-Location
Test-Path .\pyproject.toml
```

The last command should print `True`. Then run, one line at a time:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e ".[test]"
& .\.venv\Scripts\python.exe --version
& .\.venv\Scripts\python.exe -m pip check
```

The first line creates an isolated environment. The second installs MuCITE and test dependencies and may require network access. Editable installation (`-e`) is recommended for the GUI and Field Baker. A healthy dependency check reports `No broken requirements found.`

Installation also creates two console commands. Use `mucite` for the command-line
runtime and `mucite-gui` for the graphical workbench:

```powershell
& .\.venv\Scripts\mucite.exe --help
& .\.venv\Scripts\mucite.exe --config configs/headless_smoke.json
& .\.venv\Scripts\mucite-gui.exe
```

They are equivalent to `python -m src` and `python -m src.render.gui`.

You do not need to activate the environment or change PowerShell execution policy. Use `.venv\Scripts\python.exe` explicitly. Do not recreate or reinstall it on every run. For runtime-only installation, `pip install -e .` is sufficient, but `[test]` makes troubleshooting easier.

### 2.4 On the original development computer

If the verified parent project already contains `.python-runtime`, enter the release directory and use it instead:

```powershell
Set-Location "D:\users\fasma16\Projects\Simu_IonSource\github_release"
& ..\.python-runtime\python.exe --version
& ..\.python-runtime\python.exe -m src --config configs/headless_smoke.json
& ..\.python-runtime\python.exe -m src.render.gui
```

In the remainder of this guide, substitute `..\.python-runtime\python.exe` for `.\.venv\Scripts\python.exe`. Do not mix two Python environments for installation and execution.

### 2.5 Linux and headless systems

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/mucite --config configs/headless_smoke.json
```

The Tk GUI requires a graphical desktop and Tk support. Use the CLI first on servers. Local release validation used Windows CPU and is not a claim for every Linux/GPU environment.

## 3. First run without field files

### 3.1 Minimal smoke test

From the repository root:

```powershell
& .\.venv\Scripts\python.exe -m src --config configs/headless_smoke.json
```

The first run may compile kernels. The example uses CPU, seed 7, eight weighted ion packs, and a total simulated time of `1e-9 s` (1 ns). Fragmentation is disabled and a small shared static/PIC mesh and synthetic fields are used.

Local validation ended with `max_total_time`, eight survivors, 107 collisions, and 148 microsteps. Exact floating-point details may vary by platform. First confirm normal completion without a traceback or `numerical_failure`. `max_total_time` means the requested simulated duration was reached; it is not an error.

### 3.2 Generate readable results

```powershell
& .\.venv\Scripts\python.exe -m src --config configs/headless_smoke.json --progress --report --report-dir outputs/first_run
```

Open `outputs\first_run` and inspect:

- `simulation_report.md`: human-readable report.
- `simulation_summary.json`: structured summary and effective settings.
- `final_particles.csv`: final particle rows.
- `terminal_events.csv`: exit/loss events; a very short run may contain only the header.

No exit event does not imply failure. Even an unaccelerated ion moving at 250 m/s needs about 0.24 ms to travel 60 mm, far longer than the 1 ns demo.

### 3.3 Save snapshots

```powershell
& .\.venv\Scripts\python.exe -m src --config configs/headless_smoke.json --export-every 1 --export-dir outputs/first_snapshots --report --report-dir outputs/first_snapshots --no-snapshot-plots
```

`--export-every 1` means one snapshot per macro step, not one image per second. Disabling automatic plots reduces overhead but preserves numeric snapshots. Use new output directories to avoid overwriting prior results.

## 4. Using the GUI

### 4.1 Launch

```powershell
& .\.venv\Scripts\python.exe -m src.render.gui
```

The desktop GUI and CLI use the same simulation core, but a new GUI project does not duplicate every value in `headless_smoke.json`; do not require identical collision counts.

### 4.2 Window map

| Area | Purpose |
| --- | --- |
| `Project` menu | New, load, save, exit |
| `Config` menu | Beam, Field, PIC, Collision dialogs |
| `Help` menu | Parameters, workflow, author/software information |
| `Project` panel | Name, backend, output directory, field file, status |
| `Go / Stop / Generate Report` | Run, request safe stop, report the completed run |
| `Beam Setup` | Ion/source settings and initial preview |
| `Field Baker` | Import/bake fields, load a field, show alignment |
| `Runtime Setup` | Time, RF, PIC, collisions, boundaries, output; scroll down |
| Plot tabs | Snapshot, XY, RZ, phase space, temperature, energy, current, etc. |
| Lower tabs | Logs, warnings, validation, run summary, terminal information |

### 4.3 Recommended first GUI run

1. Select `Project → New`.
2. Name it `my_first_gui_run`; set output to `outputs/my_first_gui_run`.
3. Select **cpu** as the backend.
4. Keep `Use dummy static field` enabled. Do not bake or diagnose a real field yet.
5. Set Total time to `1e-9`, Macro step to `5e-10`, and seed to `7`.
6. Set Snapshot/export every to `1`, enable snapshots, and leave H5 off.
7. Set RF frequency and RF Vpeak to `0`. Leave PIC scale at 1 and grid `(0,0)`.
8. In `Config → Collision`, choose `explicit`, `iict-lite`, and fragmentation handling `off`.
9. Open `iict-lite parameters...`, select `configs/iict_lite_parameters.example.json`, leave overrides blank, and Apply both dialogs.
10. Open Beam, keep packet mode and eight packs, then Apply.
11. Run `Preview / Smoke Test` and wait for completion. This checks the initial beam, not full transport physics.
12. Press `Go`. If warned that the Beam test has not run, return and run it first.
13. Inspect Snapshot, RZ, Run Summary, and `Open Output Directory`.
14. Use `Project → Save` to retain the settings.

`Save snapshot plots` controls automatic figures, not numeric snapshots. `Auto-refresh` controls display updates, not integration time steps.

### 4.4 Stopping safely

Stop is a cooperative request handled at safe checkpoints; it is not guaranteed to terminate instantly. Wait for status/log updates before launching another task. Do not repeatedly press Go, kill several worker threads, or switch CPU/GPU during a run. If a CUDA context error occurs, close the process normally, restart it, and reproduce on CPU.

Stop does not create a resumable checkpoint. Project Save and snapshot browsing do not resume an interrupted trajectory.

### 4.5 Browser fallback

If `tkinter` is unavailable, MuCITE can fall back to a local browser GUI. To request it explicitly:

```powershell
& .\.venv\Scripts\python.exe -c "from src.render.gui import launch_main_window; launch_main_window(force_backend='web')"
```

Open the printed `http://127.0.0.1:.../` address if the browser does not open automatically. Keep the terminal open and stop it with `Ctrl+C`. Do not expose this local service to the public internet. The detailed click sequence above describes the Tk interface; the browser layout differs.

## 5. Saving, loading, and moving computers

### 5.1 Three incompatible JSON document types

| Type | Identifying structure | Consumer |
| --- | --- | --- |
| GUI project | `schema_version: 5`, with `app_config` | GUI Project Save/Load |
| CLI run document | `schema_version: 1`, with `simulation`, `ion`, optionally `execution`, `output` | `python -m src --config ...` |
| IICT parameters | `schema_version: 1`, with `heat_capacity`, `pseudoatom_mass`, `fragmentation` | IICT dialog or `--iict-parameter-config` |

`configs/headless_smoke.json` cannot be opened by GUI Project Load. A GUI Save cannot be passed directly to `--config`. Do not change only `schema_version` to convert formats. None contains live particle state or resumes a run.

Some older GUI documents can be migrated explicitly while preserving their old backend behavior. A legacy request for bundled IonSPA fails clearly in the public release rather than silently changing to iict-lite.

### 5.2 What GUI Save does and does not store

Save stores Beam, Field Baker, Runtime, Output settings, and file paths. It does not embed baked fields, raw SIMION/Fluent files, masks, external IICT JSON/CSV, running particle/RNG/GPU state, reports, snapshots, or images. “Save the project settings” is not a single-file backup of all project assets.

### 5.3 Moving to another computer

1. Copy/extract the complete release; do not copy `.venv`, `.python-runtime`, or caches.
2. Reinstall Python 3.11 and dependencies.
3. Copy data you are permitted to use: fields, masks, IICT JSON/CSV, and retained results.
4. Load the GUI project and reselect Output, Baked field, Field Baker inputs, mask, and IICT paths.
5. Replace obsolete absolute paths such as `D:\...`.
6. Run the smoke test and a short version of your case before a long run.

Always launch from the repository root and use absolute paths for data outside it. Merely placing a GUI JSON beside data does not rebase every path. One exception is IICT table `csv_path` inside an IICT JSON: it is resolved relative to that IICT JSON; direct GUI/CLI CSV overrides resolve from the project root.

## 6. Units and essential terminology

### 6.1 Units

| Unit | Meaning and warning |
| --- | --- |
| s, ms, us, ns | `1 ms=1e-3 s`, `1 us=1e-6 s`, `1 ns=1e-9 s` |
| m, mm | `1 mm=1e-3 m` |
| Hz | cycles per second; `650 kHz=650000 Hz` |
| K | absolute temperature; 300 K is about 26.85 °C |
| Pa, mbar | `1 mbar=100 Pa`; Field Baker uses Pa |
| A, nA | `1 nA=1e-9 A` |
| Da/amu | mass of one physical ion, not a pack's total mass |
| eV | energy per ion, not voltage or temperature |
| m² | collision cross section; `1 Å²=1e-20 m²` |
| J/K/ion | heat capacity per ion, not J/mol/K |

`1e-9` means 0.000000001. Enter only values in fields, not text such as `0.15 mm`. GUI fields often use mm/deg, while internal JSON/code often uses m/rad; read suffixes carefully.

### 6.2 Weighted ion packs

A weighted ion pack is one computational marker representing several real ions. One hundred packs of weight 20 represent 2,000 ions. Its position, velocity, single-ion mass, and charge define motion; weight contributes to charge density, current, and counting. It is not a molecule with twenty times the physical mass.

User-facing text says “weighted ion pack”; legacy API/JSON names such as `macro_particle_weight` remain for compatibility. A **macro step** is a time-integration interval, not a type of ion pack.

### 6.3 Three different temperatures

- Gas temperature `Tg`: thermal motion of neutral gas.
- Source temperature: input for initial translational velocity spread.
- Ion internal temperature `Ti`: internal energy used by IICT and fragmentation.

Increasing axial velocity does not directly set internal temperature. Electric fields directly change center-of-mass translation; the explicit collision model transfers internal energy.

## 7. Beam and ion source

Open `Config → Beam`; press Apply to commit edits.

### 7.1 Ion properties and population

| Parameter | Meaning |
| --- | --- |
| Ion name | Report label only; no molecular database lookup occurs |
| Mass [amu] | Mass of one ion; must be positive |
| Charge state [e] | Positive integer charge, `q=Z e` |
| Pack count | Available computational slots |
| `macro_particle_weight` | Real ions represented by each packet-mode pack |
| Collision cross section [m²] | Effective ion-neutral collision area |
| Initial internal temperature [K] | Initial internal, not translational, temperature |
| Number of atoms | Template atom count; IICT-specific values may override it |
| Heat-capacity profile, Delta H/S | Historical IonTemplate/IonSPA fields, not iict-lite inputs |

Names such as `demonstration_ion` or profile `drug` do not load physical properties automatically. The public release has no hidden profile database.

### 7.2 Packet versus continuous current

| Source mode | Behavior |
| --- | --- |
| `packet` | Launch one initial population and track it |
| `continuous-current` | Inject charge over time and reuse freed slots |

For continuous current, the real-ion rate is approximately `dN/dt = I/(Z e)`. Current is in amperes; entering `1` means 1 A, not 1 nA. Slot capacity, maximum pack weight, and packs per injection affect the numerical representation. Increasing pack weight is not a substitute for convergence in pack count.

### 7.3 Position and launch direction

Initial x/y/z define the birth center. Beam radius defines the transverse extent. `uniform-disk` and `gaussian` are alternative transverse profiles; Gaussian sigma is not the outer radius. Position/velocity jitter adds random spread. Direction and cone half-angle define static launch geometry, with `0 ≤ angle < 90`.

Do not place particles in metal. Preview the source, then compare it with Field Diagnostics.

### 7.4 Static or gas-field birth velocity

`static` means manually specified initialization; it does not require zero velocity. Initial KE, source temperature, axial speed, direction, and cone settings participate in source initialization. Preview the resulting distribution rather than assuming these quantities are interchangeable.

`from-gas-field` samples velocity at each birth location and hides/ignores incompatible static KE, source temperature, jitter, and angle fields. Ion internal temperature remains independent.

| Field | Meaning |
| --- | --- |
| Birth gas CSV | Dedicated birth-velocity data; blank lets the runtime use available baked gas |
| Birth gas z min/max [mm] | Sample range; blank max means `None`, not zero |
| Birth gas radius [mm] | Gas-sample radial filter, distinct from Beam radius |
| Radial velocity scale | 1 preserves radial flow; 0 suppresses it |
| Velocity delta [m/s] | Signed axial correction after sampling |

A dedicated birth CSV and Field Baker Fluent data are not automatically the same file. Coordinates, units, and ranges must agree.

The standalone Beam Preview receives only Beam settings and cannot infer the main window's baked field, so gas-import preview requires an explicit Birth gas CSV. The full simulation may still use baked gas when this field is blank. When preview z max is blank it uses the Beam capillary exit as its sampling upper bound, whereas the full runtime preserves `None`; specify a common range for strict comparison.

### 7.5 Capillary and prefill

Beam Capillary exit z is the runtime exit. Capillary prefill length represents an initial reservoir; zero disables it. Runtime Prefill weighted packs controls its numerical representation, with zero allowing derivation. Packs per injection and maximum ions per pack apply to continuous injection.

Field Baker also has a capillary exit for field/gas geometry. The two settings should agree with your data but are not automatically synchronized.

## 8. Field Baker, gas, and electrode boundaries

### 8.1 What baking means

Field Baker does not create electrodes or run CFD. It aligns and interpolates existing potential/gas exports onto a common grid and writes `baked_fields.npy` for fast runtime loading.

```text
SIMION DC potential + normalized RF basis + optional Fluent gas
                         ↓ Field Baker
                    baked_fields.npy
                         ↓ runtime
                    ion transport
```

Static gas means a uniform background gas. It does not mean no SIMION potential, and it is not the same as a runtime dummy field.

### 8.2 Using an existing baked field

1. Select it through Load Existing Baked Field or the Baked field Browse button.
2. Disable `Use dummy static field`.
3. Configure the corresponding electrode mask.
4. Run Show Diagnostics.
5. Check source/detector positions and RF normalization before a short run.

Load only trusted project artifacts: historical NumPy field/mask formats may contain pickled objects.

### 8.3 Baking raw inputs

| Setting | Meaning |
| --- | --- |
| SIMION DC/RF | Supported patxt/CSV potentials; RF is normally normalized +1/−1 V basis |
| Gas static | Uniform background temperature [K] and pressure [Pa] |
| Gas import | Fluent CSV plus background/fallback values |
| z/r min/max [mm] | Output grid extent; maxima must exceed minima |
| dz/dr [mm] | Grid spacing; smaller consumes more memory/time |
| PA grids per mm | Source SIMION sampling density, not output spacing |
| DC voltage scale | Multiplier applied to imported DC potential |
| Output npy/plots | Baked artifact and diagnostic destinations |

CSV headers/units must match supported import formats; the extension alone is insufficient. Advanced users can inspect [field_baker.py](../src/data/tools/field_baker.py).

Apply the dialog, run Bake, and confirm completion and the loaded-field path. Changing input temperature, pressure, or grid does not mutate an old NPY; rebake and reload.

### 8.4 Coordinates and z offsets

| SIMION | MuCITE/Python |
| --- | --- |
| x, Vx | z, vz (axial) |
| y, Vy | x, vx |
| z, Vz | y, vy |

The Fluent z offset is applied explicitly, with no automatic capillary-derived shift: conceptually `z_global = z_input + offset_z`, after confirming source units. If a Fluent outlet is at 101 mm and should map to global 6 mm, the arithmetic offset is −95 mm; use that only if your source coordinates actually match this example.

GUI presets include capillary exit 6 mm, total Fluent capillary length 101 mm, radius 0.25 mm, and external start 4.5 mm. These are interface presets, not universal instrument facts. For the development project's `slens100x_*.patxt`, the capillary region is already included and SIMION offset is 0; the public release does not include those files.

### 8.5 Electrode mask

The mask identifies metal. A potential field alone is not a complete collision boundary.

| Setting | Meaning |
| --- | --- |
| Electrode mask | Geometry source |
| Mask cache | Reusable cache tied to the same geometry/grid |
| Mask z offset | Blank/None follows baked metadata; 0 is explicit zero |
| Hit distance | Surface tolerance that can alter loss counts |

`default` may refer to historical development geometry not included publicly. Do not disable the mask simply to bypass a missing file in a real electrode problem.

### 8.6 Show Diagnostics

The GUI invokes `src.data.diagnostics.plot_field_flow_mask_alignment`, draws the global `(r,z)` range in the Snapshot area, and writes an image and `.summary.json`. Check capillary exit, gas direction, field/mask/gas alignment, source outside metal, and domain coverage. A drawable plot proves loading, not physical correctness. The dummy demo has no offline baked field for this diagnostic.

## 9. Runtime, PIC, and RF

### 9.1 Time scales

- Total time: elapsed time in the simulated world.
- Macro step: interval between space-charge field updates.
- Microstep: adaptive particle/collision step constrained by collision, RF, acceleration, spatial limits, and macro end.
- Max wall time: real computer-time limit; zero disables it.

Snapshot interval and current bin width are not integration steps. Increasing total time from 1 ns to 1 ms multiplies simulated duration by one million; increase gradually.

### 9.2 PIC

PIC deposits weighted-particle charge on a mesh, solves electrostatic potential/field, and gathers the self field back to particles.

| Setting | Meaning |
| --- | --- |
| Space-charge scale 0 | Disable self-field force only |
| Scale 1 | Physical solved amplitude |
| Other nonnegative scale | Deliberate diagnostic scaling, not an accuracy level |
| PIC nr=nz=0 | Share the static mesh |
| Both positive and ≥3 | Request exact PIC node counts; decoupled only if shape differs |

One count cannot be zero while the other is positive. Counts are nodes, not millimeters. A `2001×6501` shared mesh has about 13 million nodes before accounting for multiple arrays and solver memory. A coarser explicit mesh requires convergence validation.

### 9.3 AMG settings

Keep the beginner defaults: `amg`, `preconditioned_cg`, `ruge_stuben`, tolerance `1e-6`, 150 maximum iterations, warm start enabled. Smaller tolerance usually costs more. Maximum iterations do not guarantee convergence. Inspect the reported actual backend, convergence flag, and residual. Retained SOR modules/fields are compatibility artifacts, not an instruction to configure old SOR parameters.

### 9.4 RF and boundaries

```text
E_total(t) = E_DC + E_RF_basis × (Vpeak/Vref) × cos(2π f t + phase)
             + scale × E_space_charge
```

RF input is single-phase **Vpeak**, not Vpp. Frequency is Hz. GUI/CLI phase is degrees; internal `rf_phase_rad` is radians. Avoid scaling an RF basis that already contains the intended physical amplitude. The historical SIMION +90° versus Python −90° mapping applies only to its validated private dataset.

Detector z/radius define accepted crossings; radial limit terminates particles outside the domain. These affect statistics and are not plot limits. Capillary voltage is a capillary boundary potential, not a global DC multiplier.

## 10. Collisions, internal energy, and fragmentation

### 10.1 Two independent selections

| Selection | Options | Purpose |
| --- | --- | --- |
| Collision mode | explicit / hybrid-langevin | Scheduling/approximation strategy |
| Single-collision physics | iict-lite / ionspa | Physics used after an explicit event is selected |

Start with explicit + iict-lite. Hybrid-Langevin has a z window, switch probability, and maximum dt. It is not guaranteed to be numerically equivalent to explicit collisions and does not prove internal-energy thermalization. Batch size controls computational chunking, not collision cross section or count.

### 10.2 Minimal IICT document

```json
{
  "schema_version": 1,
  "heat_capacity": {"type": "classical", "num_atoms": 72},
  "pseudoatom_mass": {"type": "constant", "mass_da": 30.0},
  "fragmentation": {"type": "none"}
}
```

This is a runnable format example. Neither 72 atoms nor 30 Da is universal.

### 10.3 Override order

MuCITE reads IICT JSON first, then applies explicit SimulationConfig/GUI/CLI overrides. Blank GUI overrides mean “use JSON.” Atom count resolves as: dedicated IICT override, then IICT JSON, then Beam/IonTemplate only if both are absent. Changing Beam atoms does not override an explicit JSON value. Inspect `effective_parameters` and `parameter_sources` in backend runtime information.

### 10.4 Heat-capacity models

| Model | Input/formula |
| --- | --- |
| classical | `N>2`; `dof=3N−6`, `U=dof kB T` |
| constant_cv | Positive `Cv` in J/K/ion; `U=Cv T` |
| tabulated | CSV `T[K]` with either `U[J/ion]` or `Cv[J/K/ion]` |

The classical model is approximate and does not automatically reproduce quantum vibrational heat capacity. User formulas are never executed with `eval`.

Illustrative U table:

```csv
T[K],U[J/ion]
0,0
300,9e-19
600,1.8e-18
900,2.7e-18
```

Temperatures and U must be strictly increasing; U is nonnegative; Cv is positive; all values are finite. A row at T=0 must have U=0. The implementation uses piecewise-linear U/T interpolation and endpoint-slope extrapolation; Cv tables are trapezoidally integrated. Cover the actual temperature range rather than relying on large extrapolation.

### 10.5 Pseudo-atom mass

This is an effective IICT energy-transfer mass, not ion mass and not an added atom. Use either positive constant Da or a complete rectangular 2D table:

```csv
T[K],relative_speed[m/s],mass[Da]
300,100,30
300,500,32
600,100,31
600,500,33
```

Provide at least 2×2 unique nodes and every combination. Values must be finite; masses positive and inside configured bounds. Out-of-range T/speed is clamped to the table boundary, which is a numerical rule, not external physical validation.

### 10.6 Two fragmentation switches and an important limitation

The molecular-rate model is `none` or `eyring`. Eyring requires explicit ΔH‡ [kJ/mol] and ΔS‡ [J/mol/K]; it does not silently reuse Beam's legacy IonSPA fields.

```text
k(T) = (kB T/h) × exp(ΔS‡/R − ΔH‡/(R T))
P(dt) = 1 − exp(−k(T)dt)
```

Runtime product handling is separately `off`, `transport`, or `loss`. Off applies no loss. Transport reduces parent weight while retaining total transported weight; it does not generate a full fragment mass spectrum. Loss reduces transport weight and may deactivate the pack.

**Current full-runtime limitation:** this release consumes fragmentation probability during selected explicit-collision updates. It does not independently integrate hazard for every surviving hot ion at every microstep. Therefore it cannot yet establish quantitative collision-independent lifetimes, and hybrid translation does not establish internal thermalization. Beginner demos use both rate `none` and handling `off`.

### 10.7 Optional IonSPA

Only users with a separately obtained, lawful, API-compatible provider should attempt:

```powershell
& .\.venv\Scripts\python.exe -m src --config configs/headless_smoke.json --collision-physics-backend ionspa --ionspa-backend local
```

This does not install IonSPA. Missing providers fail explicitly; bundled is rejected; approximate is a historical compatibility approximation. iict-lite requires none of this. MuCITE-owned code is distributed under GPLv3; see the [license](../LICENSE.md) and [third-party notices](../THIRD_PARTY_NOTICES.md). Those terms do not grant rights to a separately supplied IonSPA provider.

## 11. Command line and configuration

Run entry points from the repository root; do not double-click internal `.py` files.

```powershell
& .\.venv\Scripts\python.exe -m src --help
& .\.venv\Scripts\python.exe -m src.data.tools.field_baker --help
```

Options are entry-point specific. For example, the runtime uses `--rf-peak-voltage`, while an alignment diagnostic uses `--rf-peak-voltage-v`.

Override one setting:

```powershell
& .\.venv\Scripts\python.exe -m src --config configs/headless_smoke.json --iict-pseudoatom-mass-da 35 --report --report-dir outputs/mass35_demo
```

Write the final canonical run configuration without running:

```powershell
& .\.venv\Scripts\python.exe -m src --config configs/headless_smoke.json --random-seed 11 --write-config configs/my_run.json
```

Run it later:

```powershell
& .\.venv\Scripts\python.exe -m src --config configs/my_run.json --report --report-dir outputs/my_run
```

JSON rules: use double quotes; numbers must not include units; booleans are `true/false`, null is `null`; comments and trailing commas are invalid. Prefer forward-slash Windows paths such as `D:/MyData/file.npy`, or escape backslashes. Unknown fields, versions, types, and ranges are rejected.

```powershell
& .\.venv\Scripts\python.exe -m json.tool configs/my_run.json
```

This checks JSON syntax only, not physical validity. Common runtime options include `--config`, `--backend`, `--total-time`, `--macro-time-step`, `--ion-count`, `--random-seed`, `--macro-particle-weight`, `--static-field`, `--dummy-static-field`, RF options, `--pic-space-charge-scale`, collision/IICT options, snapshot options, `--report`, and `--write-config`. Not every internal field has a CLI flag; do not invent one by prefixing a JSON key with `--`.

## 12. Results, snapshots, and current

| Output | Contents |
| --- | --- |
| `simulation_report.md` | Human-readable settings and statistics |
| `simulation_summary.json` | Structured effective settings/results |
| `final_particles.csv` | Final particle rows |
| `terminal_events.csv` | Exit/loss events, possibly empty |
| `snapshots_index.csv` | Snapshot times and files |
| `snapshot_*.npy` | Numeric snapshot arrays |
| `snapshots.h5` | HDF5 snapshot container |
| `metadata.json` | Snapshot columns and units |
| `representative_trajectories.csv` | Sampled track histories |
| `.png` | Configured plots/exports |

Read `metadata.json` before interpreting array columns. Optional HDF5 support is installed with:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".[hdf5]"
```

Beginners should use NPY; current live GUI delivery directly reads NPY and H5 does not imply an identical live-display path.

GUI tabs show transverse XY, axial RZ, phase space, internal temperature, kinetic energy, current, and other diagnostics. Display paths downsample large arrays (live NPY delivery may use at most 50,000 rows), so visible points are not the total pack/ion count. Export Figure saves the current plot, not all numeric results.

For a time bin with real-ion weight `N`, charge state `Z`, and width `Δt`:

```text
I_bin = N Z e / Δt
```

For `N=100000`, `Z=1`, and `Δt=0.2 ms`, current is about 0.0801 nA. Do not substitute computational pack count unless every pack has weight 1. Current bin width does not alter integration.

Inspect termination reason, final time/steps, actual backend/fields/RF/PIC, Poisson convergence/residual, source/exit/loss accounting, effective IICT sources, and warnings. Survivors are packs still active at the end, not successfully transmitted ions. With continuous sources or fragmentation, always define numerator, denominator, and time window.

## 13. From demo to research

Recommended order:

1. Preserve an unchanged public smoke test.
2. Add your field/mask and verify coordinates, RF normalization, and geometry.
3. Add justified ion, source, gas, and collision-cross-section values.
4. Add supported IICT heat/pseudo-atom data; keep fragmentation off.
5. Run a small, short case and inspect immediate losses/failures.
6. Increase packs and duration while monitoring statistics.
7. Converge microstep limit, PIC macro step, mesh, and pack population independently.
8. Only then sweep current, RF, voltage, or pressure, writing separate outputs.

There is no universal duration after which RK4 becomes inaccurate. RK4 is non-symplectic; evaluating RF at RK4 stages and freezing PIC per macro step resolves different time scales but does not guarantee energy conservation. Compare baseline, half, and quarter steps for transport, TOF, energy, temperature, current, and charge accounting. Random collision paths may change when step/sampling changes even with a fixed seed; compare statistics and multiple seeds.

RF, collisions, and injection physically change energy, so nonconstant KE alone is not a numerical-error test. Use conservative-field tests separately from open, coupled systems.

The release passed 280 tests, a short CPU smoke, GUI construction, and installed-wheel execution. These checks do not calibrate your case. Private historical SIMION comparisons were not reproduced from this public repository.

## 14. Troubleshooting

| Symptom | Action |
| --- | --- |
| Python/`py` missing or wrong version | Install 64-bit 3.11 and reopen the terminal |
| `No module named src` | Return to repository root and install editable package |
| Relative-import error | Use `python -m ...`; do not execute an internal module directly |
| Missing `configs/default.json` | Copy the complete release, not only src; specify smoke config |
| JSON parsing/unknown schema field | Identify the three JSON types; validate syntax with json.tool |
| Bundled/missing IonSPA | Select iict-lite only by explicit modeling decision, or supply lawful local provider |
| Missing IICT mass/type/ΔH | Restore a complete JSON or provide complete explicit overrides |
| Missing baked field on Go | Demo: enable dummy; research: select your real field |
| Missing `slens100x` while baking | Public release omits it; choose your own input |
| Diagnostics fail | Confirm trusted baked field, format, and mask; skip for dummy demo |
| Empty/static snapshot view | Use NPY, every=1 for short runs, Auto-refresh, and inspect index |
| No report | Enable report or complete a run before Generate Report |
| Zero exit current | Check duration, source, geometry, and detector before changing weight |
| CUDA context error | Restart the process and reproduce on CPU; CPU tests do not validate GPU |
| AMG failure/memory pressure | Reduce only for diagnosis; inspect residuals and reconverge production mesh |
| Project fails after moving | Save stores paths, not assets; copy/reselect external files |

When requesting help, provide Python version, release/path, full command or GUI project, complete traceback ending, summary if available, and whether real fields were used.

## 15. One-page checklist

For each run: enter repository root; use the correct 3.11 environment; load the intended GUI/CLI configuration; check field, mask, IICT, and output paths; check dummy/real field, CPU/GPU, PIC scale, collisions, and fragmentation; run short first; inspect warnings/report; retain configuration and results.

```powershell
# Installation smoke test
& .\.venv\Scripts\python.exe -m src --config configs/headless_smoke.json

# GUI
& .\.venv\Scripts\python.exe -m src.render.gui

# Run and report
& .\.venv\Scripts\python.exe -m src --config configs/headless_smoke.json --report --report-dir outputs/first_run
```

Next: [Source Code Responsibilities and API Guide](Source_Guide_en.md).
