"""Reusable parameter, workflow, and attribution text for the MuCITE GUI."""

from __future__ import annotations

from collections.abc import Mapping


PROJECT_PARAMETER_HELP = {
    "session_name": "Project name used to identify this run in labels, logs, JSON filenames, and generated artifacts; it must not be empty.",
    "output_dir": "Root directory for this project's snapshots, reports, figures, field-bake products, and other generated artifacts.",
    "loaded_baked_field_path": "Baked .npy electric/gas field loaded by the runtime. It is independent of the Field Baker output path until a newly baked field is selected.",
}

BEAM_PARAMETER_HELP = {
    "ion_name": "Human-readable ion or analyte name stored in reports and terminal-event records.",
    "mass_amu": "Mass of one physical ion in unified atomic mass units; the runtime converts it to kilograms for acceleration and kinetic energy.",
    "charge_e": "Positive charge state in elementary-charge units, so the physical charge is q = charge_e * e.",
    "particle_count": "Number of available weighted ion pack slots. Packet mode initially fills them; continuous-current mode reuses free slots as ions enter.",
    "source_mode": "Source lifecycle: packet launches one initial population; continuous-current injects charge over time from current_a.",
    "macro_particle_weight": "Number of physical ions represented by each packet-mode weighted ion pack; it must be positive.",
    "current_a": "Electrical ion current in amperes. It controls continuous-source ion injection and must be positive in continuous-current mode.",
    "kinetic_energy_ev": "Optional initial translational kinetic energy in eV for static velocity initialization; speed follows v = sqrt(2 E / m).",
    "initial_internal_temperature_k": "Initial internal ion temperature in kelvin used by the collision, heating, and fragmentation models.",
    "collision_cross_section_m2": "Ion-neutral collision cross section in square metres used to determine the collision rate.",
    "num_atoms": "Ion-template atom count. IonSPA uses it directly; iict-lite classical heat capacity uses it only when the dedicated IICT atom-count override is blank.",
    "heat_capacity_profile": "IonSPA heat-capacity profile identifier. It is not silently reused as an iict-lite model selection.",
    "delta_h_kj_per_mol": "Positive IonTemplate/IonSPA activation enthalpy ΔH in kJ/mol. iict-lite Eyring parameters use separate explicit fields.",
    "delta_s_j_per_mol_k": "IonTemplate/IonSPA activation entropy ΔS in J/(mol K); it is not silently reused by iict-lite.",
    "beam_radius_mm": "Outer source radius in millimetres for the transverse birth-position distribution.",
    "initial_x_mm": "Nominal initial Cartesian x coordinate in millimetres.",
    "initial_y_mm": "Nominal initial Cartesian y coordinate in millimetres.",
    "initial_z_mm": "Nominal initial axial z coordinate in millimetres.",
    "initial_position_jitter_mm": "Non-negative random position spread around the nominal source coordinates, in millimetres.",
    "velocity_jitter_m_per_s": "Non-negative random velocity spread in m/s; ignored when velocity is imported from a gas field.",
    "direction_axis": "Initial static-velocity direction, one of ±x, ±y, or ±z; gas-field initialization uses the imported direction instead.",
    "cone_half_angle_deg": "Half-angle of the static launch cone in degrees, constrained to 0 ≤ angle < 90; ignored for gas-field initialization.",
    "source_profile": "Transverse birth profile: uniform-disk or gaussian.",
    "source_gaussian_sigma_mm": "Positive Gaussian transverse standard deviation in millimetres; used only for the gaussian source profile.",
    "gas_velocity_init_mode": "Velocity source: static uses KE/temperature/axial settings; from-gas-field samples velocity at each ion birth position.",
    "source_temperature_k": "Optional positive source-gas temperature in kelvin for static initialization; omitted in gas-field mode.",
    "source_axial_velocity_m_per_s": "Optional non-negative axial gas/source speed in m/s for static initialization; omitted in gas-field mode.",
    "source_birth_velocity_gas_csv": "Optional Fluent CSV sampled specifically for birth velocities. When blank, the baked runtime gas field supplies available local data.",
    "source_birth_velocity_z_min_mm": "Lower axial bound in millimetres for the birth-velocity gas sampler; non-negative.",
    "source_birth_velocity_z_max_mm": "Optional upper axial bound in millimetres for the birth-velocity gas sampler. Blank/None means no explicit upper bound.",
    "source_birth_velocity_radius_mm": "Optional positive radial cutoff in millimetres for birth-gas samples. Blank/None keeps all available radii.",
    "source_radial_velocity_scale": "Non-negative multiplier applied to imported radial gas velocity; 0 suppresses radial flow and 1 preserves it.",
    "source_velocity_delta_m_per_s": "Signed axial velocity correction in m/s added after sampling the imported birth gas field.",
    "capillary_exit_z_mm": "Runtime axial position of the capillary exit in millimetres; continuous-source particles become externally active here.",
    "capillary_prefill_length_mm": "Length in millimetres of the steady-state capillary reservoir represented at t = 0; 0 disables prefill.",
    "notes": "Free-form project notes saved with the GUI configuration; they do not change the physics.",
}

FIELD_BAKE_PARAMETER_HELP = {
    "simion_dc_path": "SIMION DC potential source (.patxt/CSV) used to construct the static electric field.",
    "simion_rf_path": "SIMION normalized RF basis potential source (.patxt/CSV), normally representing adjacent electrodes at +1 V and -1 V.",
    "gas_field_mode": "Gas model baked into the grid: static uses background pressure/temperature; import reads spatial Fluent data.",
    "fluent_path": "Fluent gas-field CSV to import when gas_field_mode is import; required in that mode.",
    "coordinate_mapping": "Recorded source-to-global coordinate convention for the baked artifact; changing it requires physics revalidation.",
    "offset_z_simion_mm": "Explicit axial translation of SIMION data in millimetres. Current slens100x data already includes the capillary region, so use 0 mm.",
    "offset_z_fluent_mm": "Explicit axial translation applied directly to Fluent coordinates in millimetres; no automatic capillary-derived shift is added.",
    "pa_grids_per_mm": "Effective SIMION potential-array sampling density in grid units per millimetre.",
    "z_min_mm": "Minimum global axial coordinate of the baked (r,z) grid in millimetres.",
    "z_max_mm": "Maximum global axial coordinate of the baked grid in millimetres; it must exceed z_min_mm.",
    "r_min_mm": "Minimum radial coordinate of the baked axisymmetric grid in millimetres.",
    "r_max_mm": "Maximum radial coordinate of the baked axisymmetric grid in millimetres; it must exceed r_min_mm.",
    "dz_mm": "Positive axial node spacing of the baked grid in millimetres; smaller values increase resolution, memory, and bake time.",
    "dr_mm": "Positive radial node spacing of the baked grid in millimetres; smaller values increase resolution, memory, and bake time.",
    "dc_voltage_scale": "Multiplier applied to imported SIMION DC potential before electric-field derivatives are evaluated.",
    "background_pressure_pa": "Non-negative uniform gas pressure in pascals used in static gas mode and as a background/fallback value.",
    "background_temperature_k": "Positive uniform gas temperature in kelvin used in static gas mode and as a background/fallback value.",
    "capillary_exit_z_mm": "Field-baker capillary-exit coordinate in millimetres used to align/clip imported gas geometry; distinct from the runtime beam exit setting.",
    "fluent_capillary_total_length_mm": "Positive total Fluent capillary length in millimetres retained as source-geometry metadata.",
    "fluent_capillary_radius_mm": "Positive inner Fluent capillary radius in millimetres used for gas-domain geometry handling.",
    "fluent_capillary_external_start_z_mm": "Global axial coordinate where external Fluent gas data begins; it must be below the field-baker capillary exit.",
    "phi_key_for_plot": "Potential-array key selected for field-bake diagnostic plots, for example phi_dc_v.",
    "output_npy": "Destination .npy path for the baked artifact containing stable grid, simion, and fluent groups.",
    "output_plot_dir": "Directory receiving field-bake diagnostic figures and associated summaries.",
}

RUNTIME_PARAMETER_HELP = {
    "backend": "Particle execution backend: taichi uses the accelerated runtime; cpu selects the supported CPU path where available.",
    "total_time_s": "Positive simulated duration in seconds.",
    "macro_dt_s": "Positive macro-step duration in seconds. Charge is deposited and the PIC field is solved once per macro step, then frozen during its micro steps.",
    "random_seed": "Seed for reproducible physical random draws; named diagnostic streams are derived separately from it.",
    "snapshot_every": "Macro-step interval between snapshot exports; 0 disables periodic snapshot export at the runtime layer.",
    "pic_space_charge_scale": "Non-negative multiplier on the solved PIC space-charge field. 0 disables its force; 1 uses the physical solved amplitude.",
    "collision_physics_backend": "Single-event collision physics: ionspa preserves the historical optional adapter; iict-lite selects the independent paper-driven approximation without importing IonSPA.",
    "ionspa_backend": "Optional IonSPA provider: local requires a separately obtained lawful installation; approximate is the legacy project approximation. Bundled requests are rejected in this public distribution.",
    "iict_parameter_config_path": "Optional strict iict-lite schema-v1 parameter JSON. Relative paths resolve from the project root; blank means the GUI must provide a complete direct model set.",
    "iict_heat_capacity_model": "Optional iict-lite heat-capacity override: classical, constant_cv, or tabulated. Blank/None reads the type from the parameter JSON.",
    "iict_num_atoms": "Optional atom-count override for classical U = dof kB T. Blank uses the Beam/IonTemplate atom count and records that provenance.",
    "iict_constant_cv_j_per_k_per_ion": "Optional positive constant heat capacity in J/K/ion, required for a direct constant_cv model unless supplied by JSON.",
    "iict_heat_capacity_csv_path": "Optional T,U or T,Cv CSV for the tabulated heat model. A direct GUI path resolves from the project root; a JSON-relative path resolves from the JSON directory.",
    "iict_pseudoatom_model": "Optional pseudo-atom mass override: constant or tabulated. Blank/None reads the type from the parameter JSON.",
    "iict_pseudoatom_mass_da": "Optional positive constant pseudo-atom mass in daltons. No protein/N2-derived mass is inserted as a universal GUI default.",
    "iict_pseudoatom_csv_path": "Optional CSV containing temperature [K], relative speed [m/s], and mass [Da] for two-dimensional interpolation.",
    "iict_pseudoatom_min_mass_da": "Optional positive lower safety bound for pseudo-atom mass in daltons.",
    "iict_pseudoatom_max_mass_da": "Optional positive upper safety bound for pseudo-atom mass in daltons; it must exceed the lower bound.",
    "iict_fragmentation_model": "iict-lite molecular rate model: none or eyring. This is separate from fragmentation_mode, which controls how products are handled.",
    "iict_delta_h_kj_per_mol": "Optional positive Eyring activation enthalpy ΔH‡ in kJ/mol; required for direct Eyring fragmentation unless supplied by JSON.",
    "iict_delta_s_j_per_mol_k": "Optional finite Eyring activation entropy ΔS‡ in J/(mol K); required with ΔH‡ and may be negative.",
    "collision_mode": "Collision algorithm: explicit samples resolved collisions; hybrid-langevin switches to the Langevin approximation in its configured region.",
    "fragmentation_mode": "Fragment handling: transport continues products, loss records/removes them, and off disables fragmentation.",
    "rf_frequency_hz": "Non-negative RF frequency f in hertz.",
    "rf_peak_voltage_v": "Runtime single-phase RF peak voltage Vpeak in volts, not peak-to-peak voltage Vpp.",
    "rf_phase_deg": "RF phase angle in degrees. The validated SIMION export labelled +90° currently aligns with Python -90°.",
    "pic_grid_nr": "PIC radial node count. Use 0 together with pic_grid_nz=0 to share the static mesh; explicit grids require both counts ≥ 3.",
    "pic_grid_nz": "PIC axial node count. Use 0 together with pic_grid_nr=0 to share the static mesh; explicit grids require both counts ≥ 3.",
    "pic_poisson_backend": "Axisymmetric Poisson solver backend, including AMG and supported sparse direct/iterative variants.",
    "pic_poisson_preconditioner": "Optional preconditioner for compatible sparse Poisson solves: none or jacobi.",
    "pic_poisson_warm_start": "When enabled, initialize a solve from the previous macro step's potential to improve convergence.",
    "pic_amg_mode": "AMG execution mode: direct multigrid solve or AMG-preconditioned CG/BiCGSTAB.",
    "pic_amg_solver": "AMG hierarchy construction: Ruge-Stuben or smoothed aggregation.",
    "pic_amg_tolerance": "Positive relative convergence tolerance for the AMG/iterative Poisson solve.",
    "pic_amg_max_iters": "Positive maximum iteration count for the AMG/iterative Poisson solve.",
    "pic_amg_fallback_backend": "Sparse backend used if the selected AMG path is unavailable or cannot complete as configured.",
    "detector_z_mm": "Axial detector plane in millimetres; values above 0 enable transmission detection at that plane.",
    "detector_radius_mm": "Active detector radius in millimetres; values above 0 limit accepted detector crossings.",
    "radial_limit_mm": "Maximum allowed particle radius in millimetres before radial-domain termination; values above 0 enable it.",
    "capillary_voltage_v": "Electric potential assigned to the capillary boundary in volts; signed values are allowed.",
    "capillary_prefill_macro_particles": "Requested number of weighted ion packs representing the initial capillary reservoir; 0 lets the runtime derive it.",
    "macro_particles_per_injection": "Positive minimum number of weighted ion packs created for each continuous-current injection event.",
    "max_macro_particle_weight": "Maximum number of real ions represented by one continuous-source weighted ion pack; 0 disables this cap.",
    "stop_active_fraction_below": "Optional active-particle fraction threshold in [0,1] for stability-based early stopping; 0 disables the fraction trigger.",
    "stop_stable_window_steps": "Number of macro steps in the stability window for early stopping; 0 disables the window.",
    "stop_stable_fraction_tol": "Non-negative allowed change in active fraction across the stability window.",
    "max_wall_time_s": "Maximum real elapsed runtime in seconds; 0 means no wall-clock limit.",
    "terminal_event_mode": "Terminal-row retention: transport-only, all terminal causes, or none. Aggregate accounting remains separate.",
    "max_terminal_event_rows": "Maximum retained detailed terminal-event rows; 0 selects the runtime's unbounded/compatibility behavior.",
    "collision_batch_size": "Collision gather batch size. 0 selects automatic/full-batch behavior; positive values bound temporary work.",
    "langevin_z_start_mm": "Inclusive lower axial coordinate in millimetres of the hybrid Langevin region.",
    "langevin_z_end_mm": "Upper axial coordinate in millimetres of the hybrid Langevin region; it cannot precede the start in hybrid mode.",
    "langevin_switch_prob": "Hybrid switch probability in [0,1] controlling selection of the Langevin approximation in its region.",
    "langevin_max_dt_s": "Positive maximum micro-step duration in seconds while the hybrid Langevin constraint is active.",
    "electrode_mask_path": "2D electrode-mask source path; default resolves the validated S-lens geometry, while blank disables the 2D mask.",
    "electrode_mask_cache_path": "Optional cached mask artifact used to avoid reparsing/rebuilding the electrode geometry.",
    "electrode_mask_z_offset_mm": "Optional explicit mask axial offset in millimetres. Blank/None follows baked-field metadata.",
    "electrode_hit_distance_mm": "Non-negative distance tolerance in millimetres around metal surfaces; 0 uses metal-cell membership only.",
    "dummy_static_field": "Use a synthetic zero/static test field instead of loading the baked 2D field; intended for smoke tests, not validated physics runs.",
    "static_field_3d_path": "Optional Cartesian 3D external-field artifact superimposed through the supported 3D field path.",
}

OUTPUT_PARAMETER_HELP = {
    "save_snapshots": "Write periodic particle snapshots when snapshot_every permits them.",
    "save_csv": "Write supported tabular run outputs in CSV format.",
    "save_h5": "Write supported run data in HDF5 format when the required dependency and sink are available.",
    "save_figures": "Generate and retain configured diagnostic/report figures.",
    "trajectory_sample_count": "Maximum number of trajectories selected for detailed plotting; 0 disables trajectory samples.",
    "trajectory_record_every": "Positive micro/record stride used when retaining sampled trajectory points.",
    "snapshot_plot_max_points": "Positive display cap for snapshot scatter points; rendering downsamples without changing simulation state.",
    "terminal_time_bin_ms": "Positive time-bin width in milliseconds for terminal current/ion summaries, commonly 0.2 or 1 ms.",
    "report_format": "Non-empty requested report format name; markdown is the standard human-readable output.",
    "generate_report_after_run": "Automatically generate the configured report artifacts after a successful simulation.",
}

PARAMETER_HELP_BY_SECTION: Mapping[str, Mapping[str, str]] = {
    "Project": PROJECT_PARAMETER_HELP,
    "Beam / Ion": BEAM_PARAMETER_HELP,
    "Field Baker": FIELD_BAKE_PARAMETER_HELP,
    "Runtime": RUNTIME_PARAMETER_HELP,
    "Output": OUTPUT_PARAMETER_HELP,
}


SIMULATION_WORKFLOW_TEXT = """MuCITE simulation workflow

1. Configure and validate
The GUI collects project, beam, field, runtime, and output values. Saving a GUI JSON document records configuration only; transient progress and SessionState are not part of the project file. All internal numerical quantities are converted to SI units before the model runs.

2. Bake or load fields
SIMION DC/RF potentials are converted to a 2D axisymmetric (r,z) grid. Optional Fluent pressure, temperature, and velocity data are aligned with the explicit Fluent z offset. Static gas mode instead uses the configured uniform pressure and temperature. The current slens100x PA already covers z=0..64.99 mm, including its 0..5 mm empty capillary region, so its validated SIMION offset is 0 mm.

3. Create particles and source injection
Particles live in 3D Cartesian space. Static launch speed follows v = sqrt(2 E_kin / m). Continuous current represents physical charge with weighted ion packs; a capillary reservoir may be prefilled before t=0. Gas-field birth mode samples local velocity inside the configured z/r bounds; a missing z maximum means no explicit upper bound.

4. Macro step: solve space charge
At each macro step, charge carried by active weighted ion packs is deposited on the axisymmetric PIC grid. The solver evaluates ∇²φ_sc = -ρ/ε₀ and E_sc = -∇φ_sc. A PIC grid request of (0,0) shares the baked static grid; positive nr/nz values request an explicit grid. pic_space_charge_scale multiplies E_sc, with 0 disabling its force and 1 using the solved amplitude.

5. Micro steps: gather, collide, and push
For each adaptive micro step, the existing Poisson scheduler gathers local electric and gas state and evaluates collision probability P = 1 - exp(-λ Δt). An already-triggered event is then delegated to the selected single-collision physics backend: the optional IonSPA adapter or independent iict-lite implementation. iict-lite follows the cited IICT equations (DOI 10.1016/j.ijms.2024.117290) but does not claim IonSPA numerical equivalence. The runtime then advances position and velocity with RK4 using dx/dt = v and m dv/dt = q E_total. Electrode, detector, radial, and domain crossings are localized continuously along the step.

6. Fields and coordinate conventions
For 2D axisymmetric gathering, r = sqrt(x²+y²), E_x = E_r x/r, E_y = E_r y/r, and Cartesian E_z is the axial E_z. The RF field is

E(t) = E_dc + E_rf_basis (Vpeak/Vref) cos(2π f t + phase) + pic_space_charge_scale E_sc.

Vpeak is a single-phase peak voltage, not Vpp; the normalized reference is normally Vref=1 V peak. The validated empirical convention maps SIMION export +90° to Python rf_phase_deg=-90°. The fixed coordinate mapping is SIMION x→Python z, SIMION y→Python x, SIMION z→Python y, with velocity components mapped the same way.

7. Record and report
Snapshots are sampled at the requested cadence and may be display-downsampled without changing the underlying simulation. Terminal events are aggregated into current and ion counts per configured millisecond bin. Final reports record configuration, termination statistics, solver diagnostics, and selected trajectories. Rendering and reporting are output-layer services; the simulation engine can run without them.
"""


ABOUT_TEXT = """MuCITE

ESI-MS API/S-lens ion-trajectory simulation and analysis workbench.

Developed by Dr. Yihui Yan.

Open-source and licensing notice
MuCITE is free software licensed under the GNU General Public License version 3 (GPLv3). You may redistribute and modify it under that license's terms. See LICENSE.md for the complete license text. This license applies to MuCITE-owned code only; it does not grant rights to separately supplied datasets, external programs, or optional providers.

Collision-backend notice
iict-lite is an independent paper-driven approximation and neither imports IonSPA nor reads IonSPA data files when selected. The optional IonSPA compatibility adapter is retained only for users who have a lawful provider/source and must not be treated as an included license grant or as numerically equivalent to iict-lite.

Third-party notice
MuCITE uses and/or interfaces with third-party components, including Python packages, Taichi, scientific Python libraries, SIMION-derived user data, Fluent-derived user data, and optional IonSPA providers. Each component, dataset, and external tool remains subject to its own copyright, license, citation, and usage terms. Project attribution does not replace those notices.

No-warranty notice
This research software is provided as-is, without warranty of correctness, fitness for a particular purpose, uninterrupted operation, or suitability for instrument control, clinical, safety-critical, or regulatory decisions. Validate parameters, numerical convergence, field alignment, and results independently for each scientific use.
"""


def parameter_reference_text() -> str:
    """Return a readable reference containing every persistent GUI field."""

    lines = ["MuCITE parameter reference", ""]
    for section, entries in PARAMETER_HELP_BY_SECTION.items():
        lines.extend((section, "-" * len(section)))
        lines.extend(
            f"{name}\n    {description}\n" for name, description in entries.items()
        )
    return "\n".join(lines).rstrip() + "\n"


def simulation_workflow_text() -> str:
    """Return the model execution order and governing conventions."""

    return SIMULATION_WORKFLOW_TEXT.strip() + "\n"


def about_text() -> str:
    """Return authorship, source availability, and limitation notices."""

    return ABOUT_TEXT.strip() + "\n"


__all__ = [
    "ABOUT_TEXT",
    "BEAM_PARAMETER_HELP",
    "FIELD_BAKE_PARAMETER_HELP",
    "OUTPUT_PARAMETER_HELP",
    "PARAMETER_HELP_BY_SECTION",
    "PROJECT_PARAMETER_HELP",
    "RUNTIME_PARAMETER_HELP",
    "SIMULATION_WORKFLOW_TEXT",
    "about_text",
    "parameter_reference_text",
    "simulation_workflow_text",
]
