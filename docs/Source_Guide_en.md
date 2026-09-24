# MuCITE Source Code Responsibilities and API Guide

Applies to the public `src/` tree, checked on 2026-09-14. This guide explains where code belongs, how layers call one another, and where to start when changing a feature. Run the software first with the [beginner user guide](User_Guide_en.md). The generated [Package Layout](Package_Layout.md) lists all 242 Python modules, responsibilities, sizes, and duplicate-file results.

## 1. Vocabulary

| Term | Meaning | Example |
| --- | --- | --- |
| module | Related code in a file/package | `core/micro_step.py` |
| function | One operation | `load_config_document()` |
| class | Related state and behavior | `FlowchartPicSimulation` |
| dataclass | Mostly a typed parameter/state container | `SimulationConfig` |
| backend | An implementation behind a common interface | iict-lite / optional IonSPA |
| Protocol | Required operations without fixing implementation | `CollisionPhysicsBackend` |
| event | Notification that something occurred | `MacroStepCompleted` |
| adapter | Converts one representation/interface to another | GUI mm to internal m |

Leading-underscore functions are normally implementation details. User scripts should prefer public entry points instead of manually chaining private stepping methods.

## 2. Four layers and the actual tree

```text
src/
├── __main__.py          CLI entry point
├── core/                Control: time, scheduling, events, cancellation, seeds
├── env/                 Model: fields, gas, PIC, transport, collisions, sources, boundaries
├── agents/              Model: ion entities and weighted-pack state
├── config/              Configuration: models, JSON, validation, presets
├── data/                I/O: field input/baking, snapshots, reports, diagnostics
├── render/              Presentation: CLI, Tk/web GUI, optional monitor
└── utils/               Framework-independent geometry/math helpers
```

The boundaries are practical, not a claim that layers never import each other. `core/simulation.py` is the composition root and must connect model and data services. `SimulationEngine` coordinates the time loop through a runtime protocol. `data` includes input/preprocessing as well as output. Headless execution avoids GUI creation, but the CPU path still uses Taichi CPU kernels.

## 3. Call and time-flow diagrams

```text
python -m src
 → src/__main__.py → render/cli/runner.py:run_cli()
 → argument/config validation → FlowchartPicSimulation
 → SimulationEngine.run() → SimulationResult → optional data/report output

python -m src.render.gui
 → Tk or local web GUI → GUI AppConfig validation/unit conversion
 → simulation_tasks background worker → same simulation and engine
 → queue messages update presentation; data layer persists output
```

Actual time order:

```text
initialize ions/source and initial boundaries; RunStarted
macro start: stop/stage checks and macro end
PIC: deposit charge → solve Poisson → freeze self field for macro interval
micro loop:
  sample fields/gas; prepare rate and timestep candidates
  select global valid microstep
  explicit collision selection or hybrid-Langevin branch
  update collided velocity/internal temperature and current fragmentation handling
  upload changed states; RK4 push with RF sampled at its stage times
  classify boundaries; advance capillary/continuous sources; update time/stop/progress
macro end: synchronize, summarize, snapshot, MacroStepCompleted
repeat or build result/final snapshot; RunFinished; close sinks
```

This release does **not** contain a new all-active-ion fragmentation-hazard pass after collision updates. Do not infer that behavior from archived source discussions, and do not claim hybrid translational updates thermalize internal energy.

## 4. `core`: control responsibilities

| File | Responsibility |
| --- | --- |
| [simulation.py](../src/core/simulation.py) | Compose `FlowchartPicSimulation`; Taichi ownership/init/release; run request |
| [engine.py](../src/core/engine.py) | Main lifecycle, initialization, loop, finalization |
| [macro_step.py](../src/core/macro_step.py) | Macro plan, PIC refresh, history/output, stable stopping |
| [micro_step.py](../src/core/micro_step.py) | Timestep, collision→push→boundary/source ordering |
| [timestep.py](../src/core/timestep.py) | Select valid active timestep; diagnose invalid candidates |
| [run_state.py](../src/core/run_state.py) | Per-run clock, counters, termination, performance |
| [policy.py](../src/core/policy.py), [metrics.py](../src/core/metrics.py) | Progress/history retention and bounded metrics |
| [events.py](../src/core/events.py) | Lifecycle events and synchronous EventBus |
| [ports.py](../src/core/ports.py) | Runtime, progress, snapshot, and terminal-sink contracts |
| [randomness.py](../src/core/randomness.py) | SeedManager and named random streams |
| [cancellation.py](../src/core/cancellation.py), [concurrency.py](../src/core/concurrency.py) | Cooperative cancellation and run re-entry guard |
| [result_builder.py](../src/core/result_builder.py) | Final SimulationResult and completion event |

`FlowchartPicSimulation(...)` composes a run but does not advance it. `simulation.run()` delegates to the engine and returns a result. `prepare_macro_step()` refreshes PIC and returns its endpoint; `advance_micro_steps()` updates particles to that endpoint; `finalize_macro_step()` records and checks stopping. `request_cancel()` is the supported safe-stop request. Do not call Taichi reset/sync from a foreign thread; runtime release must occur on its owner thread.

## 5. `env`: model and physics

### 5.1 Fields and gas

[axisymmetric_grid.py](../src/env/fields/axisymmetric_grid.py) stores the `(r,z)` field grid; [local_sampler.py](../src/env/fields/local_sampler.py) combines external, self, and gas state at particle positions; [cartesian3d.py](../src/env/fields/cartesian3d.py) supports optional Cartesian fields. File parsing belongs to `data/fields`; sampling a loaded field belongs here. Coordinate/interpolation changes require trajectory revalidation.

[gas/layered.py](../src/env/gas/layered.py) represents gas state/layers; [velocity_sampler.py](../src/env/gas/velocity_sampler.py) samples axisymmetric flow and birth velocity. Bulk velocity is three-dimensional, and thermal velocity is random motion around it. Neither local `vz` nor laboratory translational KE defines ion internal temperature.

### 5.2 PIC

| Path | Responsibility |
| --- | --- |
| [pic/operator.py](../src/env/pic/operator.py) | Connect particles, deposition, grids, solve, field |
| [pic/poisson/factory.py](../src/env/pic/poisson/factory.py) | Select configured solver |
| [pic/poisson/amg.py](../src/env/pic/poisson/amg.py) | AMG and preconditioned iterative paths |
| [pic/poisson/sparse.py](../src/env/pic/poisson/sparse.py) | Sparse direct/iterative implementations |
| [pic/poisson/sor.py](../src/env/pic/poisson/sor.py) | Retained legacy SOR, not current GUI recommendation |
| `base.py`, `types.py` | Shared structures and solve result contracts |

PIC solves `∇²φ_sc = −ρ/ε0` and `E_sc = −∇φ_sc`. Charge density uses represented real-ion weight. `core/macro_step.py` controls refresh timing.

### 5.3 Transport

| File | Responsibility |
| --- | --- |
| [pusher.py](../src/env/transport/pusher.py) | Compose Taichi push components |
| [rk4.py](../src/env/transport/rk4.py) | Runtime RK4 kernel |
| [field_gather.py](../src/env/transport/field_gather.py) | Local interpolation and RF modulation |
| [time_step.py](../src/env/transport/time_step.py) | Kernel timestep candidates/limiters |
| [collision_rate.py](../src/env/transport/collision_rate.py) | Mean relative-speed and rate formula |
| [adaptive_integrator.py](../src/env/transport/adaptive_integrator.py) | Adaptive-RK4 helpers/settings, not the sole runtime push site |
| `initialization.py`, `terminal_classifier.py` | Initial state and termination support |

`rk4.py` implements, in simplified form, `dx/dt=v` and `dv/dt=(q/m)E_total(x,t)`, plus optional `drag_frequency_hz (u_gas−v)`. Physical **drag** is unrelated to the heat-profile name **drug**.

```text
k1=F(y,t)
k2=F(y+dt*k1/2,t+dt/2)
k3=F(y+dt*k2/2,t+dt/2)
k4=F(y+dt*k3,t+dt)
y_new=y+dt*(k1+2k2+2k3+k4)/6
```

Stage-time RF sampling resolves oscillatory force better than a start-only sample, but RK4 remains non-symplectic. Test RK4 truncation and frozen-PIC macro error separately.

### 5.4 Collisions

Separate event probability, single-event physics, and state/weight write-back.

| Path | Responsibility |
| --- | --- |
| [protocol.py](../src/env/collisions/protocol.py) | `CollisionPhysicsBackend` contract |
| [factory.py](../src/env/collisions/factory.py) | Lazily import only selected backend |
| `operator.py`, `runtime.py` | Connect collision operations to full runtime |
| `runtime_preselection.py`, `selection.py`, `probability.py` | Schedule/select explicit events |
| `explicit.py`, `hybrid.py`, `update_helpers.py` | Explicit/hybrid updates and state/weight helpers |
| [iict_lite/backend.py](../src/env/collisions/iict_lite/backend.py) | Protocol implementation; scalar/batch share core |
| [iict_lite/physics.py](../src/env/collisions/iict_lite/physics.py) | 3D single-collision/energy core with equation/DOI comments |
| `iict_lite/parameters.py` | Strict JSON/CSV resolution and provenance |
| `iict_lite/heat_capacity.py`, `pseudoatom.py`, `fragmentation.py` | Thermodynamics, effective mass, rate models |
| `ionspa_compat.py`, `adapter.py`, `backend.py` | Optional external provider bridge; bundled rejected publicly |
| `single.py`, `batch.py`, `thermo.py`, `cache.py` | Historical project adapters/helpers, not vendored IonSPA |

Scheduling is based on a rate like `ν=nσ<v_rel>` and `P=1−exp(−νdt)`. Even zero bulk flow has gas thermal motion. The backend contract supplies `energy_from_temperature` (K→J/ion), inverse conversion, pseudoatom mass (K and m/s→kg), scalar/batch already-triggered collision updates, fragmentation rate (s⁻¹), and JSON-safe `runtime_info`.

`apply_collision()` must not resample whether the collision occurs or invoke a provider's whole-step position/electric-field method. Preserve this boundary for new backends. Never copy IonSPA code/data into the public implementation to force matching.

### 5.5 Sources, boundaries, and runtime assembly

`sources/continuous.py` defines continuous current; `continuous_runtime.py` injects it; `packet_runtime.py` initializes packets; `slot_runtime.py` handles free/reused/reservoir slots; `runtime.py` and `access.py` integrate source rules. Slots can be reused, so `slot_id` is not globally unique; reconstruct trajectories by `track_id`.

`boundaries/mask2d.py` and `mask3d.py` query geometry; `electrodes.py` and `composite.py` combine boundaries; `localization.py` locates crossings within a step; `events.py` and `runtime.py` apply termination; `accounting.py` records exit/loss and parent/fragment weights. File parsing/cache belongs to `data/boundary_masks`. Detector, hit-distance, or offset changes require validation.

`env/runtime/bootstrap/` assembles artifacts, grids, RF, birth gas, controls, and services. `cloud_io.py` transfers particle state, `fields.py` manages runtime field state, `projection.py` builds statistics/snapshots, and `stages.py` handles optional staged field/source settings. These are model-composition services, not another GUI time loop.

## 6. `agents`: entity state

[ion.py](../src/agents/ion.py) defines ion state; [taichi_cloud.py](../src/agents/taichi_cloud.py) owns kernel arrays; [slot_state.py](../src/agents/slot_state.py) handles slot/active/weight state; [events.py](../src/agents/events.py) contains terminal accounting payloads. Adding an attribute requires initialization, CPU/device synchronization, slot-reuse reset, collision handling, output schema, and tests—not merely a dataclass field.

## 7. `config`: models and validation

| File | Responsibility |
| --- | --- |
| [models.py](../src/config/models.py) | Internal SI IonTemplate, SimulationConfig, ExecutionConfig, OutputConfig |
| [json_config.py](../src/config/json_config.py) | Strict versioned ConfigDocument load/dump |
| [factories.py](../src/config/factories.py) | Normalize/construct configs |
| `validation.py`, `tuning_validation.py`, `iict_validation.py` | Types, ranges, cross-field rules |
| `contracts.py`, `constants.py` | Cross-layer result/progress contracts and constants/status codes |
| `release_defaults.py`, `presets/` | Public demo request and installed-package resources |
| `cli_adapter.py` | Retained conversion API; presentation parsing is under render/cli |

`render/gui/models.py` is different: it stores presentation values such as mm and blank optionals. It is not a duplicate of internal SI models. Direct `SimulationConfig()` retains historical defaults and may not select public demo resources; scripts should load an explicit JSON and use `dataclasses.replace`.

## 8. `data`: artifacts, records, and tools

`data/artifacts/` parses patxt and metadata schemas; `data/fields/runtime_artifacts.py` loads baked/staged fields; `data/boundary_masks/` reads mask/cache files. `data/tools/field_baker.py`, `field_baker_3d.py`, and `electrode_mask_baker_3d.py` preprocess fields/masks. Do not casually rename stable `grid`, `simion`, or `fluent` groups and coordinate/RF metadata.

[logger.py](../src/data/logger.py) writes NPY/H5 snapshots, index, trajectories, and metadata. `terminal_*` modules define, stream, sample, and summarize terminal rows. `data/report/snapshot.py` captures immutable report data; summary modules calculate results; writer/Markdown/CSV/plot modules render them. Add report fields here after finding their physical source rather than changing RK4.

Useful entry-point help:

```powershell
& .\.venv\Scripts\python.exe -m src.data.tools.field_baker --help
& .\.venv\Scripts\python.exe -m src.data.diagnostics.plot_field_flow_mask_alignment --help
& .\.venv\Scripts\python.exe -m src.data.benchmarks.simion_field_benchmark --help
```

Field-bake template requiring your own files:

```powershell
& .\.venv\Scripts\python.exe -m src.data.tools.field_baker `
  --simion-dc-csv "D:/MyData/dc.patxt" `
  --simion-rf-csv "D:/MyData/rf.patxt" `
  --offset-z-simion-mm 0 `
  --z-min-mm 0 --z-max-mm 65 --r-min-mm 0 --r-max-mm 20 `
  --dz-mm 0.1 --dr-mm 0.1 --simion-pa-grids-per-mm 100 `
  --background-pressure-pa 200 --background-temperature-k 300 `
  --output-npy outputs/my_field/baked_fields.npy `
  --output-plot-dir outputs/my_field/plots
```

For Fluent add `--fluent-csv`, an explicit z offset, and matching capillary geometry. PowerShell continuation backticks must be the final character on a line.

```powershell
& .\.venv\Scripts\python.exe -m src.data.diagnostics.plot_field_flow_mask_alignment `
  --field outputs/my_field/baked_fields.npy `
  --electrode-mask "D:/MyData/electrodes.patxt" `
  --global-window --output outputs/my_field/alignment.png
```

`diagnostics/analyze_*` generally expects existing sweep data. `benchmarks/` supports pure-field SIMION comparison; strict particle-by-particle validation needs identical initial rays.

## 9. `render`: CLI and GUI

`render/cli/parser.py` parses and records explicit flags; `arguments/` groups options; `config_adapter.py` merges CLI and JSON; `runner.py` builds/runs/reports; `runtime_window.py` is an optional monitor. Adding a flag requires both definition and mapping/validation.

GUI responsibilities:

| Modules | Responsibility |
| --- | --- |
| `__init__`, `__main__`, `main_window` | Launch and top-level window |
| `models`, `config_io`, `release_migration` | AppConfig, strict project JSON, legacy behavior |
| `config_adapter`, `iict_config_validation` | Presentation-to-runtime conversion |
| `beam_dialog`, `field_dialog`, `runtime_dialogs`, `iict_dialog` | Editors |
| `window_*layout`, `window_actions`, `window_variables`, `window_state` | Layout, actions, binding/state |
| `workers`, `worker_lifecycle`, `*_tasks` | Background work/cancellation and task adapters |
| `window_results`, `plotting`, terminal modules | Result messages, figures, current bins |
| `help_content`, `window_help`, `web_*`, `cli_preview` | Help, browser UI, CLI preview |

GUI JSON excludes SessionState. Historical `session_*` identifiers do not imply resumable runs. A GUI parameter addition must cover model, widget, binding, JSON, unit conversion, CLI preview, web binding, Help, and tests.

## 10. `utils`

[geometry.py](../src/utils/geometry.py) contains framework-independent geometry helpers. Physics belongs in env and presentation helpers in render; do not use utils as an unstructured miscellaneous folder.

## 11. Python API usage

Run the supplied examples:

```powershell
& .\.venv\Scripts\python.exe -m examples.headless_json
& .\.venv\Scripts\python.exe -m examples.engine_events
```

Minimal API run with snapshots:

```python
from dataclasses import replace
from pathlib import Path
from src.config import load_config_document
from src.core.simulation import FlowchartPicSimulation, release_pic_taichi_runtime
from src.data.logger import DataLogger

document = load_config_document(Path("configs/headless_smoke.json"))
config = replace(document.simulation, random_seed=7)
logger = DataLogger(Path("outputs/api_demo"), export_every_macro_steps=1, make_plots=False)
simulation = FlowchartPicSimulation(document.ion, config, particle_backend="cpu", data_logger=logger)
try:
    result = simulation.run()
    print("Termination:", result.termination_reason)
    print("Surviving packs:", len(result.survivors))
finally:
    release_pic_taichi_runtime()
```

This does not automatically implement `document.output` CLI report policy. Use the CLI/`run_cli()` for complete CLI reporting or explicitly integrate report snapshots/writers.

Inspect IICT without transport:

```python
from pathlib import Path
from src.config import load_config_document
from src.env.collisions.factory import build_collision_physics_backend

document = load_config_document(Path("configs/headless_smoke.json"))
backend = build_collision_physics_backend(config=document.simulation, template=document.ion)
energy = backend.energy_from_temperature(300.0)
print(energy, backend.temperature_from_energy(energy))
print(backend.runtime_info)
```

Convert a saved GUI project through the real adapter rather than changing a version number:

```python
from pathlib import Path
from src.config import ExecutionConfig, OutputConfig, dump_config_document, make_config_document
from src.render.gui.config_io import load_app_config
from src.render.gui.config_adapter import build_ion_template, build_simulation_config

app = load_app_config(Path("outputs/my_first_gui_run/my_project.json"))
document = make_config_document(
    simulation=build_simulation_config(app),
    ion=build_ion_template(app.beam),
    execution=ExecutionConfig(backend=app.runtime.backend, no_window=True),
    output=OutputConfig(report=True, report_dir=Path("outputs/converted_run")),
)
dump_config_document(Path("configs/converted_run.json"), document)
```

This converts the physical request and explicit execution/report policy. It does not copy assets, bake fields, or reproduce every GUI-only display preference.

EventBus callbacks are synchronous. Queue expensive plotting/network work externally; observers should not mutate particle state or consume physical collision random streams.

## 12. Where to change a feature

| Goal | Start in | Minimum validation |
| --- | --- | --- |
| Change a case | config copy/GUI/CLI | Effective values and short run |
| Add CLI field | arguments + adapter + config | defaults, override, round-trip, invalid input |
| Add GUI field | models/dialog/binding/JSON/adapter | units, save/load, CLI/web consistency |
| Add report column | data/report + terminal schemas | provenance, units, compatibility, empty data |
| Change snapshot view | plotting/window_results | Downsampling must not alter physics |
| Change heat/effective mass | IICT JSON/CSV | units, sources, ranges, U/T round-trip |
| New collision backend | protocol/factory/new package | scalar/batch, conservation, thermalization, one scheduler |
| Fragmentation timing | micro_step + collision write-back | no-collision decay, dt/2/dt/4, once/step, off/hybrid, trajectory/count parity |
| RK4/RF | transport + micro_step | analytic force, timestep/long-run convergence, phase |
| PIC | pic + bootstrap/grids + config | residual, boundaries, sharing, macro/mesh convergence |
| Coordinates/mask/schema | data artifacts/tools + boundaries | same-ray particle validation |

Do not mechanically split physics sequencing only because a file is long; first fix behavior with tests. Existing size/duplicate findings are in Package Layout.

## 13. Tests and release tools

```powershell
& .\.venv\Scripts\python.exe -m compileall -q src tests examples tools
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe tools/gui_smoke.py
& .\.venv\Scripts\python.exe tools/check_release.py
```

Compileall checks compilation, not physics. Pytest covers distributed configuration, IICT, transport/PIC, GUI logic, and release boundaries. GUI smoke constructs hidden windows, not full GPU interaction. Release audit is not legal/security certification.

`tools/inventory.py` builds the source layout/hash inventory; `tools/package_github.py` creates the clean archive; `.github/workflows/ci.yml` defines future hosted checks. Rebuild/audit:

```powershell
& .\.venv\Scripts\python.exe tools/package_github.py
& .\.venv\Scripts\python.exe tools/check_release.py --archive dist/MuCITE-github.zip
```

Do not publish the parent project, private data, runtime environments, paper PDFs, IonSPA, or unreviewed outputs. MuCITE-owned code is distributed under GPLv3; see `LICENSE.md`. Third-party tools, optional providers, and user-supplied datasets retain their own terms.

## 14. Recommended reading order

1. `examples/headless_json.py`.
2. `config/models.py` for inputs and units.
3. `core/engine.py` for lifecycle.
4. `core/macro_step.py` and `micro_step.py` for sequencing.
5. The relevant transport, iict-lite, or PIC model package.
6. `data/logger.py` and `data/report` for output meaning.
7. GUI/CLI adapters to trace presentation values into physics.

You need not start with every `__init__.py` or historical diagnostic. Follow input → state update → output, then use [Package Layout](Package_Layout.md) for individual modules.
