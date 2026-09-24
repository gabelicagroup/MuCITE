# Package layout

Generated from module docstrings and AST.

The package name remains `src` to preserve existing module/API paths.

| Module | Responsibility | Lines | Largest function / class |
| --- | --- | ---: | ---: |
| `src/__init__.py` | Package root for the Simu_IonSource codebase. | 6 | 0 / 0 |
| `src/__main__.py` | Package entry point for python -m src. | 12 | 0 / 0 |
| `src/agents/__init__.py` | Particle/entity state for the model layer. | 36 | 10 / 0 |
| `src/agents/events.py` | Entity-level terminal event payloads and weighted accounting. | 121 | 28 / 71 |
| `src/agents/ion.py` | Particle entity state independent of transport and presentation. | 24 | 0 / 11 |
| `src/agents/slot_state.py` | Particle slot metadata storage for the coupled runtime. | 108 | 29 / 89 |
| `src/agents/taichi_cloud.py` | Taichi-backed storage for 3D ion macro-particle entities. | 126 | 33 / 120 |
| `src/config/__init__.py` | Stable public configuration API for the coupled runtime. | 125 | 0 / 0 |
| `src/config/cli_adapter.py` | Conversion of parsed CLI arguments into canonical ion requests. | 118 | 29 / 0 |
| `src/config/constants.py` | Physical constants, repository paths, and stable runtime codes. | 52 | 0 / 0 |
| `src/config/contracts.py` | Runtime result and callback contracts shared across package boundaries. | 91 | 0 / 25 |
| `src/config/factories.py` | Canonical factories for normalized and validated request models. | 78 | 13 / 0 |
| `src/config/iict_validation.py` | Focused validation for optional iict-lite configuration fields. | 94 | 20 / 0 |
| `src/config/json_config.py` | Versioned, fail-closed JSON documents for simulation and ion requests. | 276 | 41 / 8 |
| `src/config/models.py` | Immutable user-request models for ions and coupled simulations. | 190 | 0 / 122 |
| `src/config/presets/__init__.py` | Packaged demonstration requests for source and wheel installations. | 1 | 0 / 0 |
| `src/config/release_defaults.py` | Resolve the explicit public demo from a checkout or installed package. | 27 | 10 / 0 |
| `src/config/tuning_validation.py` | Strict validation for externalized runtime and retention tuning. | 116 | 34 / 0 |
| `src/config/validation.py` | Normalization and validation for canonical configuration requests. | 463 | 30 / 0 |
| `src/core/__init__.py` | Control-layer primitives for deterministic simulation execution. | 37 | 6 / 0 |
| `src/core/cancellation.py` | Thread-safe cooperative cancellation owned by the control layer. | 22 | 2 / 15 |
| `src/core/concurrency.py` | Process-local guard against concurrent reuse of one simulation runtime. | 36 | 7 / 24 |
| `src/core/engine.py` | Macro/micro event loop for the coupled simulation runtime. | 165 | 35 / 147 |
| `src/core/events.py` | Synchronous lifecycle events emitted by the simulation engine. | 52 | 9 / 19 |
| `src/core/macro_step.py` | Macro-step scheduling and bounded diagnostics. | 194 | 38 / 6 |
| `src/core/metrics.py` | Fixed-memory online statistics owned by the control layer. | 368 | 24 / 113 |
| `src/core/micro_step.py` | Micro-step scheduling for the coupled particle runtime. | 300 | 39 / 5 |
| `src/core/policy.py` | Non-physical cadence and bounded-memory policy for one engine run. | 44 | 22 / 29 |
| `src/core/ports.py` | Dependency-inversion ports used by the control layer. | 181 | 8 / 125 |
| `src/core/randomness.py` | Deterministic random-seed ownership for one simulation runtime. | 42 | 6 / 31 |
| `src/core/result_builder.py` | Build the stable result contract at the control/output boundary. | 74 | 40 / 0 |
| `src/core/run_state.py` | Mutable state for exactly one engine run. | 115 | 13 / 34 |
| `src/core/simulation.py` | Core ion-transport simulation orchestration and PIC runtime. | 304 | 45 / 72 |
| `src/core/timestep.py` | Backend-independent time-step validation and selection. | 104 | 37 / 22 |
| `src/data/__init__.py` | Output, monitoring, and persistence adapters. | 1 | 0 / 0 |
| `src/data/artifacts/__init__.py` | Artifact parsing and schema validation. | 3 | 0 / 0 |
| `src/data/artifacts/grid_metadata.py` | Strict validation for baked axisymmetric grid metadata. | 330 | 49 / 11 |
| `src/data/artifacts/patxt.py` | Strict streaming parser for retained two-dimensional SIMION PATXT files. | 466 | 40 / 0 |
| `src/data/artifacts/patxt_models.py` | Data transfer objects used by the strict PATXT parser. | 65 | 0 / 15 |
| `src/data/artifacts/schema.py` | Compatibility facade for strict retained/generated artifact schemas. | 30 | 0 / 0 |
| `src/data/benchmarks/__init__.py` | Standalone validation and comparison benchmarks. | 1 | 0 / 0 |
| `src/data/benchmarks/simion_field_benchmark.py` | Pure-electric-field trajectory benchmark against SIMION-style ion beams. | 1194 | 195 / 26 |
| `src/data/benchmarks/simion_trajectory_compare.py` | Time-matched trajectory comparison against a SIMION flight log. | 769 | 182 / 12 |
| `src/data/boundary_masks/__init__.py` | Persistence adapters for environment electrode-mask models. | 21 | 0 / 0 |
| `src/data/boundary_masks/mask2d_io.py` | File adapters for axisymmetric electrode-mask models. | 239 | 36 / 0 |
| `src/data/boundary_masks/mask3d_io.py` | File adapters for Cartesian 3D electrode-mask models. | 137 | 29 / 0 |
| `src/data/boundary_masks/metadata.py` | Serialization helpers for electrode-mask metadata. | 25 | 12 / 0 |
| `src/data/diagnostics/__init__.py` | Offline diagnostics, plotting, and reporting scripts. | 1 | 0 / 0 |
| `src/data/diagnostics/analyze_collision_model_validation.py` | Summarize collision-model validation runs and extract matched exit tracks. | 416 | 87 / 0 |
| `src/data/diagnostics/analyze_current_sweep_multimetric_per_ms.py` | Extract per-ms transport, energy, fragmentation, and loss metrics for a current sweep. | 280 | 110 / 0 |
| `src/data/diagnostics/analyze_current_sweep_validation.py` | Summarize current-sweep validation runs and extract matched exit tracks. | 439 | 92 / 0 |
| `src/data/diagnostics/analyze_rf_sweep_multimetric_per_ms.py` | Package entry/export module | 554 | 123 / 0 |
| `src/data/diagnostics/analyze_rf_sweep_validation.py` | Summarize RF-sweep validation runs and extract matched exit trajectories. | 420 | 74 / 0 |
| `src/data/diagnostics/analyze_snapshot_case.py` | Package entry/export module | 300 | 90 / 0 |
| `src/data/diagnostics/analyze_stable_terminal_summary.py` | Summarize stable-window transport, energy, fragmentation, and terminal losses. | 120 | 50 / 0 |
| `src/data/diagnostics/generate_sanity_report.py` | Generate a one-page English report summarizing the sanity-check results. | 214 | 163 / 0 |
| `src/data/diagnostics/pic_convergence.py` | PIC mesh and macro-coupling convergence diagnostics. | 488 | 91 / 16 |
| `src/data/diagnostics/plot_best_trajectory_comparison.py` | Plot the closest SIMION/Python trajectory with axial error bars. | 369 | 89 / 41 |
| `src/data/diagnostics/plot_continuous_source_diagnostics.py` | Generate diagnostic plots for completed continuous-current runs. | 264 | 48 / 0 |
| `src/data/diagnostics/plot_field_flow_mask_alignment.py` | Plot electric field, gas flow, and electrode mask alignment. | 584 | 238 / 0 |
| `src/data/diagnostics/plot_fig7_thermalization.py` | Build manuscript Fig. 7 from partial thermalization snapshot data. | 434 | 50 / 20 |
| `src/data/diagnostics/plot_fluent_thermodynamics.py` | Plot Mach number, temperature, and pressure from a Fluent axisymmetric export. | 528 | 174 / 0 |
| `src/data/diagnostics/plot_gaussian_source_diagnostics.py` | Generate Gaussian source and tube-gas birth velocity diagnostic figures. | 699 | 66 / 10 |
| `src/data/diagnostics/plot_ion_loss_on_electrode_mask.py` | Plot ion-loss terminal events on top of a 2D electrode mask. | 1015 | 128 / 15 |
| `src/data/diagnostics/plot_results.py` | Offline plotting utilities for exported macro-step particle snapshots. | 206 | 62 / 5 |
| `src/data/diagnostics/plot_simulation_workflow.py` | Generate a compact paper workflow diagram for the ion-source simulation. | 183 | 85 / 0 |
| `src/data/diagnostics/plot_style.py` | Shared paper-figure styling helpers for diagnostic plots. | 94 | 28 / 0 |
| `src/data/diagnostics/plot_vr_trajectory.py` | Compatibility entry point for local v_r trajectory diagnostic plots. | 12 | 0 / 0 |
| `src/data/diagnostics/plot_vr_trajectory_overlay.py` | Plot local gas radial velocity contours with representative ion tracks. | 1045 | 116 / 14 |
| `src/data/diagnostics/sanity_checks.py` | Physics sanity checks for the axisymmetric PIC workflow. | 419 | 94 / 0 |
| `src/data/diagnostics/summarize_trap_only_capacity.py` | Summarize trap-only PIC capacity scan outputs. | 223 | 65 / 0 |
| `src/data/diagnostics/validate_langevin_thermalization.py` | Validate the high-collision Langevin thermalization approximation. | 193 | 80 / 0 |
| `src/data/fields/__init__.py` | Field artifact readers and runtime installation adapters. | 3 | 0 / 0 |
| `src/data/fields/runtime_artifacts.py` | Data-layer field-artifact loading and staged-field helpers. | 446 | 41 / 9 |
| `src/data/logger.py` | Snapshot exporter implementing the core SnapshotSink port. | 304 | 47 / 279 |
| `src/data/report/__init__.py` | Immutable report capture and artifact writers. | 16 | 0 / 0 |
| `src/data/report/csv_tables.py` | CSV artifact writers for immutable simulation report snapshots. | 93 | 22 / 0 |
| `src/data/report/dto.py` | Report-facing DTO mappings for ion templates and simulation config. | 216 | 41 / 0 |
| `src/data/report/final_particles.py` | Final-particle table and transport summaries for reports. | 323 | 39 / 15 |
| `src/data/report/markdown.py` | Markdown rendering for the stable simulation report. | 500 | 34 / 0 |
| `src/data/report/plots.py` | Diagnostic plots generated from immutable report rows. | 323 | 36 / 0 |
| `src/data/report/runtime_summary.py` | Runtime, source-accounting, survivor, and performance report summaries. | 397 | 50 / 21 |
| `src/data/report/schema.py` | Public report-schema API assembled from focused helper modules. | 39 | 0 / 0 |
| `src/data/report/snapshot.py` | Immutable, lightweight report input captured at the runtime boundary. | 302 | 47 / 24 |
| `src/data/report/summary_builder.py` | Stable JSON summary construction from immutable report snapshots. | 180 | 47 / 0 |
| `src/data/report/terminal_summary.py` | Terminal-event accounting and bounded-distribution report summaries. | 177 | 46 / 0 |
| `src/data/report/values.py` | JSON normalization and numerical statistics for report schemas. | 77 | 24 / 0 |
| `src/data/report/writer.py` | Orchestrate report artifacts from an immutable runtime snapshot. | 110 | 44 / 0 |
| `src/data/snapshot_plots.py` | Plot adapters for persisted particle snapshots and trajectories. | 247 | 32 / 11 |
| `src/data/terminal_recorder.py` | Bounded terminal-event collection and optional CSV persistence. | 357 | 36 / 295 |
| `src/data/terminal_reservoir.py` | Bounded in-memory retention for streamed terminal events. | 58 | 19 / 48 |
| `src/data/terminal_rows.py` | Normalization and row construction for terminal-event batches. | 180 | 41 / 17 |
| `src/data/terminal_schema.py` | Stable column schema for terminal-event persistence. | 39 | 0 / 0 |
| `src/data/terminal_statistics.py` | Exact additive accounting for recorded and omitted terminal events. | 189 | 27 / 176 |
| `src/data/terminal_stream.py` | Incremental CSV stream owned by the terminal-event output layer. | 57 | 11 / 46 |
| `src/data/tools/__init__.py` | Standalone field-baking and preprocessing tools. | 1 | 0 / 0 |
| `src/data/tools/electrode_mask_baker_3d.py` | Bake regular Cartesian 3D electrode masks from CSV point exports. | 237 | 72 / 0 |
| `src/data/tools/field_baker.py` | 离线场数据烘焙器：将 SIMION / Fluent 导出场统一映射到全局 2D 轴对称网格。 | 1427 | 238 / 711 |
| `src/data/tools/field_baker_3d.py` | Bake regular Cartesian 3D electric fields for multipole ion guides. | 468 | 82 / 10 |
| `src/data/tools/field_convergence.py` | Field-baker convergence diagnostics for SIMION refined PA text fields. | 798 | 171 / 8 |
| `src/data/tools/ring_trap_pic_setup.py` | Prepare ring-electrode trap PIC validation inputs. | 350 | 74 / 9 |
| `src/env/__init__.py` | Simulation environment models and physical operators. | 1 | 0 / 0 |
| `src/env/boundaries/__init__.py` | Environment boundary policies exposed to the simulation runtime. | 26 | 10 / 0 |
| `src/env/boundaries/accounting.py` | Terminal time-of-flight, weighted accounting, and recorder adaptation. | 325 | 50 / 108 |
| `src/env/boundaries/composite.py` | Composed terminal-boundary runtime policy. | 16 | 0 / 6 |
| `src/env/boundaries/electrodes.py` | Electrode occupancy sampling and swept thin-solid detection. | 278 | 47 / 195 |
| `src/env/boundaries/events.py` | Internal event batches exchanged by terminal-boundary runtime steps. | 79 | 23 / 44 |
| `src/env/boundaries/localization.py` | Continuous localization of the earliest terminal segment event. | 354 | 48 / 51 |
| `src/env/boundaries/mask2d.py` | Axisymmetric electrode-mask model, validation, and sampling. | 222 | 32 / 7 |
| `src/env/boundaries/mask3d.py` | Cartesian 3D electrode-mask model, validation, and sampling. | 273 | 38 / 10 |
| `src/env/boundaries/runtime.py` | Runtime collection, ordering, and commit of terminal boundary events. | 314 | 39 / 264 |
| `src/env/collisions/__init__.py` | Canonical collision operators and pluggable physics backends. | 45 | 12 / 0 |
| `src/env/collisions/adapter.py` | Public deterministic bridge to bundled or approximate IonSPA. | 47 | 19 / 28 |
| `src/env/collisions/backend.py` | Deterministic bundled/approximate IonSPA backend selection. | 137 | 34 / 74 |
| `src/env/collisions/batch.py` | Vectorized IonSPA collision updates. | 199 | 46 / 38 |
| `src/env/collisions/cache.py` | IonSPA thermodynamic-model cache. | 55 | 23 / 40 |
| `src/env/collisions/constants.py` | Constants shared by collision and IonSPA adapters. | 13 | 0 / 0 |
| `src/env/collisions/explicit.py` | Explicit IonSPA collision and fragmentation updates. | 347 | 42 / 45 |
| `src/env/collisions/factory.py` | Lazy construction of collision-physics backends. | 38 | 24 / 0 |
| `src/env/collisions/hybrid.py` | Hybrid Langevin collision updates. | 106 | 35 / 38 |
| `src/env/collisions/iict_lite/__init__.py` | Independent Improved Impulsive Collision Theory backend. | 63 | 0 / 0 |
| `src/env/collisions/iict_lite/backend.py` | Independent paper-driven IICT collision backend. | 276 | 50 / 225 |
| `src/env/collisions/iict_lite/fragmentation.py` | Eyring kinetics from Prell 2024 Eq. 60-61. | 118 | 11 / 43 |
| `src/env/collisions/iict_lite/heat_capacity.py` | Independent heat-capacity models for the paper-driven IICT backend. | 257 | 29 / 64 |
| `src/env/collisions/iict_lite/parameters.py` | Strict JSON/CSV parameter resolution for the independent IICT backend. | 498 | 36 / 10 |
| `src/env/collisions/iict_lite/physics.py` | Paper-derived IICT impulse and statistical validation formulas. | 420 | 47 / 11 |
| `src/env/collisions/iict_lite/pseudoatom.py` | User-supplied pseudo-atom mass models for independent IICT. | 237 | 31 / 74 |
| `src/env/collisions/ionspa_compat.py` | Optional IonSPA compatibility adapter implementing the common protocol. | 127 | 18 / 109 |
| `src/env/collisions/models.py` | IonSPA adapter model types. | 36 | 4 / 13 |
| `src/env/collisions/operator.py` | Public whole-macro collision operator. | 46 | 18 / 28 |
| `src/env/collisions/probability.py` | Collision probability formulas. | 15 | 5 / 0 |
| `src/env/collisions/protocol.py` | Backend-neutral collision-physics contract. | 79 | 16 / 62 |
| `src/env/collisions/runtime.py` | Composed collision behavior for the main simulation runtime. | 16 | 0 / 6 |
| `src/env/collisions/runtime_common.py` | Shared collision-runtime state and dispatch methods. | 73 | 19 / 59 |
| `src/env/collisions/runtime_legacy.py` | Legacy full-gather collision runtime. | 74 | 34 / 37 |
| `src/env/collisions/runtime_preselection.py` | Cached-rate collision preselection for the Taichi runtime. | 214 | 33 / 35 |
| `src/env/collisions/selection.py` | Whole-macro collision candidate selection. | 131 | 41 / 0 |
| `src/env/collisions/single.py` | Single-ion IonSPA collision update. | 144 | 42 / 22 |
| `src/env/collisions/thermo.py` | IonSPA fragmentation and pseudo-atom thermodynamic helpers. | 145 | 28 / 128 |
| `src/env/collisions/update_helpers.py` | Sparse collision-update aggregation. | 66 | 27 / 8 |
| `src/env/fields/__init__.py` | Electric, gas, and space-charge field models. | 61 | 10 / 0 |
| `src/env/fields/axisymmetric_grid.py` | Environment-owned axisymmetric field buffers for the unified 2D (r, z) grid. | 426 | 50 / 294 |
| `src/env/fields/cartesian3d.py` | Regular Cartesian 3D electric-field loading and sampling utilities. | 279 | 40 / 100 |
| `src/env/fields/local_sampler.py` | CPU-side runtime field cache and axisymmetric gather boundary. | 330 | 39 / 229 |
| `src/env/gas/__init__.py` | Gas environment models and source-side velocity sampling. | 10 | 0 / 0 |
| `src/env/gas/layered.py` | Reduced-order layered gas dynamics for ESI-MS interface simulations. | 261 | 45 / 218 |
| `src/env/gas/velocity_sampler.py` | Source-side gas velocity sampling from raw axisymmetric Fluent exports. | 264 | 43 / 130 |
| `src/env/pic/__init__.py` | Axisymmetric particle-in-cell operators. | 30 | 10 / 0 |
| `src/env/pic/operator.py` | Environment-owned PIC charge-deposition and Poisson operator. | 180 | 49 / 158 |
| `src/env/pic/poisson/__init__.py` | Public axisymmetric Poisson solver API. | 31 | 0 / 0 |
| `src/env/pic/poisson/amg.py` | Optional pyamg-backed axisymmetric Poisson solver. | 291 | 43 / 275 |
| `src/env/pic/poisson/base.py` | Common geometry, residual, and solver interfaces for PIC Poisson backends. | 240 | 47 / 70 |
| `src/env/pic/poisson/factory.py` | Factory for axisymmetric PIC Poisson backends. | 56 | 48 / 0 |
| `src/env/pic/poisson/sor.py` | Legacy red-black SOR reference backend. | 341 | 35 / 184 |
| `src/env/pic/poisson/sparse.py` | Finite-volume sparse backends for the axisymmetric PIC Poisson equation. | 470 | 45 / 207 |
| `src/env/pic/poisson/types.py` | Shared result types and constants for axisymmetric Poisson solvers. | 28 | 0 / 15 |
| `src/env/runtime/__init__.py` | Non-collision model-runtime behavior and bootstrap composition. | 18 | 0 / 7 |
| `src/env/runtime/bootstrap/__init__.py` | Public bootstrap API for the coupled model runtime. | 10 | 0 / 0 |
| `src/env/runtime/bootstrap/artifacts.py` | Resolve immutable configuration and field artifacts before allocation. | 64 | 39 / 10 |
| `src/env/runtime/bootstrap/composition.py` | Ordered composition root for the coupled simulation runtime. | 171 | 37 / 0 |
| `src/env/runtime/bootstrap/control.py` | Install control objects, deterministic RNG, and stage metadata. | 76 | 48 / 0 |
| `src/env/runtime/bootstrap/grids.py` | Taichi backend, shared/decoupled grids, masks, and model operators. | 218 | 47 / 0 |
| `src/env/runtime/bootstrap/messages.py` | Construct the established backend/configuration status message. | 80 | 30 / 0 |
| `src/env/runtime/bootstrap/rf.py` | Validate and install PIC/RF scalar runtime state. | 99 | 43 / 0 |
| `src/env/runtime/bootstrap/services.py` | Injected outer-layer services used by the model-runtime composition root. | 48 | 0 / 38 |
| `src/env/runtime/bootstrap/source_birth.py` | Resolve optional Fluent-based source birth velocity sampling. | 127 | 38 / 6 |
| `src/env/runtime/cloud_io.py` | Taichi cloud upload/download adapters for the coupled runtime. | 139 | 30 / 129 |
| `src/env/runtime/fields.py` | Field cache, PIC macro-update, and sampling delegates. | 85 | 21 / 74 |
| `src/env/runtime/projection.py` | Progress, survivor, and particle-table projections. | 115 | 38 / 104 |
| `src/env/runtime/stages.py` | Static-field stage scheduling glue for the model runtime. | 74 | 19 / 67 |
| `src/env/sources/__init__.py` | Particle source boundary models. | 10 | 0 / 0 |
| `src/env/sources/access.py` | Small source-model accessors used by the control-layer runtime port. | 97 | 22 / 87 |
| `src/env/sources/continuous.py` | Continuous ion-current source model and capillary phase-space sampling. | 327 | 41 / 228 |
| `src/env/sources/continuous_runtime.py` | Continuous-current source lifecycle for the simulation model. | 300 | 38 / 287 |
| `src/env/sources/packet_runtime.py` | Packet-source initialization behavior for the simulation model. | 151 | 29 / 135 |
| `src/env/sources/runtime.py` | Aggregate source behavior exposed by the coupled model runtime. | 20 | 0 / 7 |
| `src/env/sources/slot_runtime.py` | Particle-slot mutations for continuous source injection. | 211 | 41 / 193 |
| `src/env/transport/__init__.py` | Particle transport strategies and composed pusher. | 30 | 10 / 0 |
| `src/env/transport/adaptive_integrator.py` | Time integration and macro-scale space-charge utilities. | 140 | 39 / 122 |
| `src/env/transport/collision_rate.py` | Taichi-compatible collision-rate formulas for the runtime scheduler. | 56 | 27 / 0 |
| `src/env/transport/field_gather.py` | Taichi field interpolation and axisymmetric-to-Cartesian mapping. | 185 | 32 / 180 |
| `src/env/transport/initialization.py` | Particle-pusher field and boundary-buffer initialization. | 188 | 32 / 0 |
| `src/env/transport/pusher.py` | Composed 3D particle transport operator. | 65 | 36 / 44 |
| `src/env/transport/rk4.py` | Taichi RK4 transport under gathered electric and optional drag fields. | 136 | 34 / 131 |
| `src/env/transport/terminal_classifier.py` | Taichi endpoint terminal-event classification. | 172 | 42 / 167 |
| `src/env/transport/time_step.py` | Taichi timestep-candidate preparation and diagnostic download. | 363 | 44 / 182 |
| `src/render/__init__.py` | CLI and GUI presentation adapters. | 1 | 0 / 0 |
| `src/render/cli/__init__.py` | Command-line presentation and request-adaptation layer. | 6 | 0 / 0 |
| `src/render/cli/arguments/__init__.py` | Argument-group registration helpers. | 21 | 0 / 0 |
| `src/render/cli/arguments/boundary.py` | Detector and early-termination CLI arguments. | 26 | 3 / 0 |
| `src/render/cli/arguments/collision.py` | Collision, IonSPA, and fragmentation CLI arguments. | 40 | 3 / 0 |
| `src/render/cli/arguments/field.py` | Static/RF field and electrode-mask CLI arguments. | 37 | 3 / 0 |
| `src/render/cli/arguments/ion.py` | Ion-template CLI arguments. | 31 | 3 / 0 |
| `src/render/cli/arguments/output.py` | Snapshot and report CLI arguments. | 26 | 3 / 0 |
| `src/render/cli/arguments/pic.py` | PIC and Poisson-solver CLI arguments. | 31 | 3 / 0 |
| `src/render/cli/arguments/runtime.py` | Process and macro-runtime CLI arguments. | 32 | 3 / 0 |
| `src/render/cli/arguments/source.py` | Ion-source and capillary CLI arguments. | 44 | 3 / 0 |
| `src/render/cli/config_adapter.py` | Adapt parsed CLI values onto a canonical versioned configuration. | 454 | 47 / 7 |
| `src/render/cli/parser.py` | CLI parser construction and explicit-option tracking. | 76 | 14 / 4 |
| `src/render/cli/runner.py` | CLI execution orchestration over canonical request documents. | 257 | 39 / 0 |
| `src/render/gui/__init__.py` | Graphical presentation front-end for the MuCITE ion-source runtime. | 34 | 25 / 0 |
| `src/render/gui/__main__.py` | Canonical entry point for the GUI presentation package. | 9 | 0 / 0 |
| `src/render/gui/beam_dialog.py` | Beam-source configuration dialog with mode-aware velocity controls. | 476 | 34 / 292 |
| `src/render/gui/beam_tasks.py` | Beam preview sampling and offline smoke-test tasks. | 314 | 49 / 0 |
| `src/render/gui/cli_preview.py` | Stable CLI argument preview generated from GUI configuration. | 344 | 43 / 9 |
| `src/render/gui/config_adapter.py` | Adapt GUI model values into canonical simulation and ion contracts. | 449 | 50 / 0 |
| `src/render/gui/config_io.py` | Strict JSON save/load helpers for GUI configuration documents. | 432 | 42 / 0 |
| `src/render/gui/dialogs.py` | Shared Tk dialog primitives and lazy compatibility exports. | 94 | 16 / 68 |
| `src/render/gui/field_baker_layout.py` | Tk widget construction for the field-baker window. | 198 | 25 / 0 |
| `src/render/gui/field_baker_ui.py` | 简易 Tkinter UI：用于检查 SIMION / Fluent 场的网格对齐并启动离线烘焙。 | 305 | 49 / 276 |
| `src/render/gui/field_dialog.py` | Field, gas, and electrode-mask configuration dialog. | 428 | 37 / 295 |
| `src/render/gui/field_tasks.py` | Offline field-baking task adapter for the GUI. | 220 | 43 / 0 |
| `src/render/gui/help_content.py` | Reusable parameter, workflow, and attribution text for the MuCITE GUI. | 250 | 10 / 0 |
| `src/render/gui/iict_config_validation.py` | Strict, presentation-layer validation for editable IICT overrides. | 123 | 38 / 0 |
| `src/render/gui/iict_dialog.py` | Editable JSON and direct overrides for the independent IICT backend. | 328 | 42 / 294 |
| `src/render/gui/main_window.py` | Compatibility facade and composition root for the Tk GUI window. | 107 | 32 / 50 |
| `src/render/gui/models.py` | Presentation configuration containers for the MuCITE GUI. | 213 | 2 / 65 |
| `src/render/gui/plotting.py` | Matplotlib figure builders for GUI presentation. | 261 | 48 / 0 |
| `src/render/gui/release_migration.py` | Preserve omitted historical JSON fields despite new-project demo defaults. | 34 | 28 / 0 |
| `src/render/gui/runtime_dialogs.py` | Focused PIC and collision configuration dialogs. | 413 | 36 / 175 |
| `src/render/gui/simulation_tasks.py` | Simulation, snapshot, and report background tasks for the GUI. | 362 | 49 / 0 |
| `src/render/gui/terminal_view.py` | Time-binned terminal current and ion-count presentation helpers. | 279 | 43 / 0 |
| `src/render/gui/web_config_binding.py` | Mapping between the browser form payload and GUI configuration DTOs. | 201 | 33 / 0 |
| `src/render/gui/web_handler.py` | HTTP transport adapter for the browser GUI. | 187 | 44 / 159 |
| `src/render/gui/web_iict_controls.py` | Browser-form fragments for collision-physics backend configuration. | 99 | 13 / 0 |
| `src/render/gui/web_messages.py` | Worker-message reducers for the browser presentation adapter. | 273 | 41 / 0 |
| `src/render/gui/web_state.py` | Mutable browser workbench state, independent of HTTP routing. | 196 | 25 / 161 |
| `src/render/gui/web_window.py` | Browser-based presentation adapter for environments without Tkinter. | 488 | 25 / 2 |
| `src/render/gui/window_actions.py` | User actions and worker dispatch for the Tk main window. | 405 | 40 / 245 |
| `src/render/gui/window_help.py` | Scrollable Tk help windows for the MuCITE workbench. | 68 | 33 / 51 |
| `src/render/gui/window_layout.py` | Main-window layout and visualization widget construction. | 357 | 30 / 299 |
| `src/render/gui/window_results.py` | Worker-message routing and result visualization for the Tk window. | 416 | 44 / 206 |
| `src/render/gui/window_runtime_layout.py` | Compact runtime controls with focused PIC and collision dialogs. | 183 | 35 / 174 |
| `src/render/gui/window_state.py` | Configuration synchronization and text state for the Tk main window. | 249 | 35 / 234 |
| `src/render/gui/window_terminal_layout.py` | Terminal-current table layout kept separate from the main window shell. | 52 | 22 / 23 |
| `src/render/gui/window_terminal_results.py` | Terminal-event source selection and Tk result presentation. | 136 | 24 / 118 |
| `src/render/gui/window_variables.py` | Tk variable construction for the main GUI window. | 65 | 22 / 58 |
| `src/render/gui/worker_lifecycle.py` | Thread lifecycle and message transport for GUI background tasks. | 62 | 31 / 4 |
| `src/render/gui/workers.py` | Compatibility facade for split GUI background-task adapters. | 52 | 0 / 0 |
| `src/render/runtime_window.py` | Tkinter presentation adapter for live simulation output. | 210 | 28 / 190 |
| `src/utils/__init__.py` | Pure reusable mathematics with no framework-layer dependency. | 15 | 0 / 0 |
| `src/utils/geometry.py` | Pure Cartesian segment geometry used by terminal-event policies. | 143 | 36 / 0 |

## Exact duplicate Python contents

No non-empty exact duplicates found.

## Size warnings

Preserved oversized modules (over 500 lines) are listed below. Packaging does
not refactor their physics or plotting logic.

- `src/data/benchmarks/simion_field_benchmark.py`: 1194 lines
- `src/data/benchmarks/simion_trajectory_compare.py`: 769 lines
- `src/data/diagnostics/analyze_rf_sweep_multimetric_per_ms.py`: 554 lines
- `src/data/diagnostics/plot_field_flow_mask_alignment.py`: 584 lines
- `src/data/diagnostics/plot_fluent_thermodynamics.py`: 528 lines
- `src/data/diagnostics/plot_gaussian_source_diagnostics.py`: 699 lines
- `src/data/diagnostics/plot_ion_loss_on_electrode_mask.py`: 1015 lines
- `src/data/diagnostics/plot_vr_trajectory_overlay.py`: 1045 lines
- `src/data/tools/field_baker.py`: 1427 lines
- `src/data/tools/field_convergence.py`: 798 lines
