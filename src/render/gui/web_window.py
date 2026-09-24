"""Browser-based presentation adapter for environments without Tkinter."""

from __future__ import annotations

import json
import webbrowser
from http.server import ThreadingHTTPServer

from .help_content import about_text, parameter_reference_text, simulation_workflow_text
from .web_handler import WebGuiHandler
from .web_iict_controls import inject_iict_controls
from .web_state import WebGuiState


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MuCITE GUI</title>
  <style>
    :root { color-scheme: light; --line: #d7dde7; --ink: #1d2733; --muted: #5a6878; --blue: #1f5eff; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Segoe UI, Arial, sans-serif; color: var(--ink); background: #f5f7fa; }
    header { height: 54px; display: flex; align-items: center; gap: 8px; padding: 0 14px; background: #fff; border-bottom: 1px solid var(--line); }
    header h1 { font-size: 17px; margin: 0 18px 0 0; }
    button { border: 1px solid #b8c2d1; background: #fff; color: var(--ink); border-radius: 4px; padding: 6px 10px; cursor: pointer; }
    button.primary { background: var(--blue); color: #fff; border-color: var(--blue); }
    button:disabled { opacity: 0.55; cursor: default; }
    nav { display: flex; align-items: center; gap: 5px; }
    .signature { margin-left: auto; color: var(--muted); font-size: 11px; }
    .menu { position: relative; }
    .menu summary { list-style: none; border: 1px solid transparent; border-radius: 4px; padding: 6px 10px; cursor: pointer; }
    .menu summary::-webkit-details-marker { display: none; }
    .menu[open] summary, .menu summary:hover { border-color: #b8c2d1; background: #f5f7fa; }
    .menu-items { position: absolute; z-index: 5; top: calc(100% + 4px); left: 0; min-width: 180px; padding: 5px; border: 1px solid var(--line); border-radius: 5px; background: #fff; box-shadow: 0 5px 18px #1d273326; }
    .menu-items button { display: block; width: 100%; border: 0; text-align: left; white-space: nowrap; }
    main { display: grid; grid-template-columns: 360px 1fr; grid-template-rows: auto 1fr 230px; height: calc(100vh - 54px); }
    .project { grid-column: 1 / 3; display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 18px; align-items: center; padding: 10px 14px; background: #fff; border-bottom: 1px solid var(--line); }
    .project-fields { display: grid; grid-template-columns: 90px minmax(140px, 1fr) 70px 120px 60px 110px; gap: 8px; align-items: center; }
    .project-actions { display: flex; gap: 10px; align-items: stretch; }
    .project-actions button { min-width: 92px; min-height: 48px; padding: 10px 14px; font-size: 14px; font-weight: 600; }
    aside { overflow: auto; padding: 10px; border-right: 1px solid var(--line); background: #fff; }
    section { border: 1px solid var(--line); border-radius: 6px; padding: 10px; margin-bottom: 10px; background: #fff; }
    section h2 { font-size: 13px; margin: 0 0 8px; }
    label { font-size: 12px; color: var(--muted); display: block; margin: 7px 0 3px; }
    input, select { width: 100%; border: 1px solid #c8d0dc; border-radius: 4px; padding: 6px 7px; background: #fff; }
    .two { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .viewer { padding: 10px; overflow: hidden; display: grid; grid-template-rows: auto 1fr; gap: 8px; }
    .tabs { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
    .canvas { min-height: 0; border: 1px solid var(--line); border-radius: 6px; background: #fff; display: flex; align-items: center; justify-content: center; overflow: hidden; }
    .canvas img { max-width: 100%; max-height: 100%; object-fit: contain; }
    .empty { color: var(--muted); }
    .console { grid-column: 1 / 3; border-top: 1px solid var(--line); background: #101722; color: #d6deea; display: grid; grid-template-columns: repeat(6, 1fr); min-height: 0; }
    .logpane { border-right: 1px solid #263244; padding: 8px; overflow: auto; font-family: Consolas, monospace; font-size: 12px; white-space: pre-wrap; }
    .logpane strong { display: block; color: #fff; margin-bottom: 4px; font-family: Segoe UI, Arial, sans-serif; }
    .statusline { font-size: 12px; color: var(--muted); margin-top: 6px; }
    .hidden { display: none !important; }
    .subgroup { border-top: 1px solid var(--line); margin-top: 9px; padding-top: 3px; }
    .terminalpane { grid-column: span 2; }
    .terminalpane table { width: 100%; border-collapse: collapse; white-space: nowrap; }
    .terminalpane th, .terminalpane td { padding: 2px 5px; text-align: right; border-bottom: 1px solid #263244; }
    dialog { width: min(760px, calc(100vw - 40px)); max-height: calc(100vh - 60px); border: 1px solid var(--line); border-radius: 8px; padding: 0; box-shadow: 0 12px 38px #10172244; }
    dialog::backdrop { background: #10172266; }
    .help-head { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--line); }
    .help-head h2 { margin: 0; font-size: 16px; }
    #help-body { margin: 0; padding: 14px; overflow: auto; white-space: pre-wrap; font: 13px/1.55 Segoe UI, Arial, sans-serif; }
  </style>
</head>
<body>
  <header>
    <h1>MuCITE GUI</h1>
    <nav aria-label="Application menu">
      <details class="menu"><summary>Project</summary><div class="menu-items">
        <button onclick="newProject()">New Project</button><button onclick="saveProject()">Save Project</button><button onclick="loadProject()">Load Project</button>
      </div></details>
      <details class="menu"><summary>Config</summary><div class="menu-items">
        <button onclick="showConfig('beam-section')">Beam Setup</button><button onclick="showConfig('field-section')">Field Baker</button><button onclick="showConfig('runtime-section')">Runtime Setup</button><button onclick="applyCurrentConfig()">Apply Config</button>
      </div></details>
      <details class="menu"><summary>Help</summary><div class="menu-items">
        <button onclick="showHelp('parameters')">Parameter Reference</button><button onclick="showHelp('workflow')">Simulation Workflow</button><button onclick="showHelp('about')">About MuCITE</button>
      </div></details>
    </nav>
    <span class="signature">Dr. Yihui Yan</span>
  </header>
  <main>
    <div class="project">
      <div class="project-fields">
        <label>Project</label><input id="session_name">
        <label>Execution backend</label><select id="runtime.backend"><option>cpu</option><option>taichi</option></select>
        <label>Status</label><strong id="status">CONFIGURED</strong>
        <label>Output dir</label><input id="output_dir" style="grid-column: span 5;">
        <label>Baked field</label><input id="loaded_baked_field_path" style="grid-column: span 5;">
      </div>
      <div class="project-actions">
        <button class="primary" onclick="postAction('/api/start/simulation')">Go</button>
        <button onclick="postAction('/api/stop')">Stop</button>
        <button onclick="postAction('/api/report')">Generate Report</button>
      </div>
    </div>
    <aside>
      <section id="beam-section">
        <h2>Beam Setup</h2>
        <div class="two">
          <div><label>Ion name</label><input id="beam.ion_name"></div>
          <div><label>Number of atoms</label><input id="beam.num_atoms" type="number"></div>
          <div><label>Mass [amu]</label><input id="beam.mass_amu" type="number" step="any"></div>
          <div><label>Charge [e]</label><input id="beam.charge_e" type="number"></div>
          <div><label>Particles</label><input id="beam.particle_count" type="number"></div>
          <div><label>Current [A]</label><input id="beam.current_a" type="number" step="any"></div>
          <div><label>Ion internal T [K]</label><input id="beam.initial_internal_temperature_k" type="number" step="any"></div>
          <div><label>Heat-capacity profile</label><input id="beam.heat_capacity_profile"></div>
          <div><label>Delta H [kJ/mol]</label><input id="beam.delta_h_kj_per_mol" type="number" step="any"></div>
          <div><label>Delta S [J/(mol K)]</label><input id="beam.delta_s_j_per_mol_k" type="number" step="any"></div>
          <div><label>CCS [m2]</label><input id="beam.collision_cross_section_m2" type="number" step="any"></div>
          <div><label>Capillary exit z [mm]</label><input id="beam.capillary_exit_z_mm" type="number" step="any"></div>
          <div><label>Capillary prefill [mm]</label><input id="beam.capillary_prefill_length_mm" type="number" step="any"></div>
          <div><label>Radius [mm]</label><input id="beam.beam_radius_mm" type="number" step="any"></div>
          <div><label>Initial z [mm]</label><input id="beam.initial_z_mm" type="number" step="any"></div>
          <div><label>Gaussian sigma [mm]</label><input id="beam.source_gaussian_sigma_mm" type="number" step="any"></div>
          <div><label>Position jitter [mm]</label><input id="beam.initial_position_jitter_mm" type="number" step="any"></div>
        </div>
        <label>Source mode</label><select id="beam.source_mode"><option>packet</option><option>continuous-current</option></select>
        <label>Source profile</label><select id="beam.source_profile"><option>uniform-disk</option><option>gaussian</option></select>
        <label>Initial velocity</label><select id="beam.gas_velocity_init_mode"><option value="static">Static values</option><option value="from-gas-field">Import from gas field</option></select>
        <div id="beam-static-fields" class="two subgroup">
          <div><label>Initial KE [eV]</label><input id="beam.kinetic_energy_ev" type="number" step="any"></div>
          <div><label>Source T [K]</label><input id="beam.source_temperature_k" type="number" step="any"></div>
          <div><label>Axial velocity [m/s]</label><input id="beam.source_axial_velocity_m_per_s" type="number" step="any"></div>
          <div><label>Velocity jitter [m/s]</label><input id="beam.velocity_jitter_m_per_s" type="number" step="any"></div>
          <div><label>Cone [deg]</label><input id="beam.cone_half_angle_deg" type="number" step="any"></div>
          <div><label>Direction</label><select id="beam.direction_axis"><option>+z</option><option>-z</option><option>+x</option><option>-x</option><option>+y</option><option>-y</option></select></div>
        </div>
        <div id="beam-gas-fields" class="subgroup">
          <label>Birth gas CSV</label><input id="beam.source_birth_velocity_gas_csv">
          <div class="two">
          <div><label>Birth z min [mm]</label><input id="beam.source_birth_velocity_z_min_mm" type="number" step="any"></div>
          <div><label>Birth z max [mm] (blank = automatic)</label><input id="beam.source_birth_velocity_z_max_mm" type="number" step="any"></div>
          <div><label>Birth radius [mm] (blank = automatic)</label><input id="beam.source_birth_velocity_radius_mm" type="number" min="0" step="any"></div>
          <div><label>Radial velocity scale</label><input id="beam.source_radial_velocity_scale" type="number" step="any"></div>
          <div><label>Axial velocity delta [m/s]</label><input id="beam.source_velocity_delta_m_per_s" type="number" step="any"></div>
          </div>
        </div>
        <div class="statusline" id="beam_summary"></div>
        <button onclick="postAction('/api/start/beam-smoke')">Beam Smoke</button>
      </section>
      <section id="field-section">
        <h2>Field Baker</h2>
        <label>SIMION DC</label><input id="field_bake.simion_dc_path">
        <label>SIMION RF</label><input id="field_bake.simion_rf_path">
        <label>Gas field</label><select id="field_bake.gas_field_mode"><option value="static">Static gas</option><option value="import">Import gas field</option></select>
        <div class="two subgroup">
          <div><label>Static / background pressure [Pa]</label><input id="field_bake.background_pressure_pa" type="number" step="any"></div>
          <div><label>Static / background temperature [K]</label><input id="field_bake.background_temperature_k" type="number" step="any"></div>
        </div>
        <div id="field-static-gas-fields" class="subgroup">
          Uniform static gas uses these pressure and temperature values with zero velocity.
        </div>
        <div id="field-import-gas-fields" class="subgroup">
          <label>Fluent gas CSV</label><input id="field_bake.fluent_path">
          <label>Fluent z offset [mm]</label><input id="field_bake.offset_z_fluent_mm" type="number" step="any">
          <div class="two">
            <div><label>Capillary total length [mm]</label><input id="field_bake.fluent_capillary_total_length_mm" type="number" step="any"></div>
            <div><label>Capillary radius [mm]</label><input id="field_bake.fluent_capillary_radius_mm" type="number" step="any"></div>
            <div><label>External gas start z [mm]</label><input id="field_bake.fluent_capillary_external_start_z_mm" type="number" step="any"></div>
          </div>
          <div class="statusline">Fluent z offset is used directly; total length does not modify it. Pressure and temperature above fill cells outside the imported gas domain.</div>
        </div>
        <div class="two">
          <div><label>z max [mm]</label><input id="field_bake.z_max_mm" type="number" step="any"></div>
          <div><label>r max [mm]</label><input id="field_bake.r_max_mm" type="number" step="any"></div>
          <div><label>dz [mm]</label><input id="field_bake.dz_mm" type="number" step="any"></div>
          <div><label>dr [mm]</label><input id="field_bake.dr_mm" type="number" step="any"></div>
          <div><label>PA grids/mm</label><input id="field_bake.pa_grids_per_mm" type="number" step="any"></div>
          <div><label>SIMION offset [mm]</label><input id="field_bake.offset_z_simion_mm" type="number" step="any"></div>
          <div><label>Capillary exit z [mm]</label><input id="field_bake.capillary_exit_z_mm" type="number" step="any"></div>
        </div>
        <div class="subgroup">
          <label>Electrode mask source</label><input id="runtime.electrode_mask_path">
          <label>Electrode mask cache</label><input id="runtime.electrode_mask_cache_path">
          <div class="two">
            <div><label>Mask z offset [mm]</label><input id="runtime.electrode_mask_z_offset_mm" type="number" step="any"></div>
            <div><label>Hit distance [mm]</label><input id="runtime.electrode_hit_distance_mm" type="number" step="any"></div>
          </div>
        </div>
        <label>Output NPY</label><input id="field_bake.output_npy">
        <button class="primary" onclick="postAction('/api/start/field-bake')">Run Bake</button>
        <button onclick="postAction('/api/field-diagnostics')">Show Diagnostics</button>
      </section>
      <section id="runtime-section">
        <h2>Runtime Setup</h2>
        <div class="two">
          <div><label>Total time [s]</label><input id="runtime.total_time_s" type="number" step="any"></div>
          <div><label>Macro dt [s]</label><input id="runtime.macro_dt_s" type="number" step="any"></div>
          <div><label>Random seed</label><input id="runtime.random_seed" type="number"></div>
          <div><label>Snapshot every</label><input id="runtime.snapshot_every" type="number"></div>
          <div><label>RF Vpeak [V]</label><input id="runtime.rf_peak_voltage_v" type="number" step="any"></div>
          <div><label>RF freq [Hz]</label><input id="runtime.rf_frequency_hz" type="number" step="any"></div>
          <div><label>RF phase [deg]</label><input id="runtime.rf_phase_deg" type="number" step="any"></div>
          <div><label>Detector z [mm]</label><input id="runtime.detector_z_mm" type="number" step="any"></div>
          <div><label>Detector radius [mm]</label><input id="runtime.detector_radius_mm" type="number" step="any"></div>
          <div><label>Radial limit [mm]</label><input id="runtime.radial_limit_mm" type="number" step="any"></div>
          <div><label>Capillary voltage [V]</label><input id="runtime.capillary_voltage_v" type="number" step="any"></div>
          <div><label>Prefill weighted ion packs</label><input id="runtime.capillary_prefill_macro_particles" type="number"></div>
          <div><label>Weighted packs per injection</label><input id="runtime.macro_particles_per_injection" type="number"></div>
          <div><label>Max ions per weighted pack</label><input id="runtime.max_macro_particle_weight" type="number" step="any"></div>
          <div><label>Max wall [s]</label><input id="runtime.max_wall_time_s" type="number" step="any"></div>
          <div><label>Max terminal rows (0 = exact/unlimited)</label><input id="runtime.max_terminal_event_rows" type="number"></div>
          <div><label>Collision batch</label><input id="runtime.collision_batch_size" type="number"></div>
        </div>
        <label>PIC space-charge scale (0 = off)</label><input id="runtime.pic_space_charge_scale" type="number" min="0" step="any">
        <div class="subgroup">
          <div class="statusline">PIC nr=nz=0 shares the static mesh; a positive pair selects an explicit decoupled mesh.</div>
          <div class="two">
            <div><label>PIC nr (0 = shared)</label><input id="runtime.pic_grid_nr" type="number" min="0"></div>
            <div><label>PIC nz (0 = shared)</label><input id="runtime.pic_grid_nz" type="number" min="0"></div>
          </div>
          <label>Poisson backend</label><select id="runtime.pic_poisson_backend"><option>amg</option><option>sparse_direct</option><option>sparse_spsolve</option><option>sparse_cg</option><option>sparse_bicgstab</option></select>
          <label>Preconditioner</label><select id="runtime.pic_poisson_preconditioner"><option>none</option><option>jacobi</option></select>
          <label><input id="runtime.pic_poisson_warm_start" type="checkbox"> Warm start Poisson solve</label>
          <div id="pic-amg-fields" class="two">
            <div><label>AMG mode</label><select id="runtime.pic_amg_mode"><option>solve</option><option>preconditioned_cg</option><option>preconditioned_bicgstab</option></select></div>
            <div><label>AMG hierarchy</label><select id="runtime.pic_amg_solver"><option>ruge_stuben</option><option>smoothed_aggregation</option></select></div>
            <div><label>AMG tolerance</label><input id="runtime.pic_amg_tolerance" type="number" step="any"></div>
            <div><label>AMG max iterations</label><input id="runtime.pic_amg_max_iters" type="number"></div>
            <div><label>AMG fallback</label><select id="runtime.pic_amg_fallback_backend"><option>sparse_direct</option><option>sparse_spsolve</option><option>sparse_cg</option><option>sparse_bicgstab</option></select></div>
          </div>
        </div>
__IICT_PANEL_HTML__
        <label>Collision</label><select id="runtime.collision_mode"><option>explicit</option><option>hybrid-langevin</option></select>
        <div id="hybrid-collision-fields" class="two subgroup">
          <div><label>Langevin z start [mm]</label><input id="runtime.langevin_z_start_mm" type="number" step="any"></div>
          <div><label>Langevin z end [mm]</label><input id="runtime.langevin_z_end_mm" type="number" step="any"></div>
          <div><label>Langevin switch prob</label><input id="runtime.langevin_switch_prob" type="number" step="any"></div>
          <div><label>Langevin max dt [s]</label><input id="runtime.langevin_max_dt_s" type="number" step="any"></div>
        </div>
        <label>Terminal events</label><select id="runtime.terminal_event_mode"><option>transport-only</option><option>all</option><option>none</option></select>
        <label>Fragment product handling</label><select id="runtime.fragmentation_mode"><option>transport</option><option>loss</option><option>off</option></select>
        <label>Trajectory sample count</label><input id="output.trajectory_sample_count" type="number">
        <label>Trajectory record every</label><input id="output.trajectory_record_every" type="number">
        <label>Snapshot plot max points</label><input id="output.snapshot_plot_max_points" type="number">
        <label>Terminal current bin [ms]</label><select id="output.terminal_time_bin_ms"><option value="0.2">0.2</option><option value="1.0">1.0</option></select>
        <label><input id="output.save_snapshots" type="checkbox"> Save snapshots</label>
        <label><input id="output.save_h5" type="checkbox"> Save as H5</label>
        <label><input id="output.save_figures" type="checkbox"> Save plots</label>
        <label><input id="output.generate_report_after_run" type="checkbox"> Generate report</label>
      </section>
    </aside>
    <div class="viewer">
      <div class="tabs">
        <button onclick="postAction('/api/field-diagnostics')">Field</button>
        <button onclick="refresh()">Refresh</button>
        <span id="progress" class="statusline"></span>
      </div>
      <div class="canvas" id="canvas"><span class="empty">No figure yet.</span></div>
    </div>
    <div class="console">
      <div class="logpane"><strong>Logs</strong><span id="logs"></span></div>
      <div class="logpane"><strong>Warnings</strong><span id="warnings"></span></div>
      <div class="logpane"><strong>Validation</strong><span id="validation"></span></div>
      <div class="logpane"><strong>Run Summary</strong><span id="run_summary"></span></div>
      <div class="logpane terminalpane"><strong>Terminal Report</strong><table><thead><tr><th>time [ms]</th><th>active weighted packs</th><th>exit [nA]</th><th>exit ions</th><th>exit weighted packs</th><th>electrode packs</th><th>other loss packs</th></tr></thead><tbody id="terminal_rows"></tbody></table></div>
    </div>
  </main>
  <dialog id="help-dialog">
    <div class="help-head"><h2 id="help-title">Help</h2><button onclick="document.getElementById('help-dialog').close()">Close</button></div>
    <pre id="help-body"></pre>
  </dialog>
  <script>
    const helpContent = __HELP_CONTENT__;
    const ids = [
      'session_name','output_dir','loaded_baked_field_path',
      'beam.ion_name','beam.mass_amu','beam.charge_e','beam.particle_count','beam.source_mode','beam.current_a',
      'beam.kinetic_energy_ev','beam.initial_internal_temperature_k','beam.num_atoms',
      'beam.heat_capacity_profile','beam.delta_h_kj_per_mol','beam.delta_s_j_per_mol_k',
      'beam.beam_radius_mm','beam.initial_z_mm','beam.capillary_exit_z_mm','beam.capillary_prefill_length_mm',
      'beam.cone_half_angle_deg','beam.direction_axis','beam.source_profile',
      'beam.source_gaussian_sigma_mm','beam.initial_position_jitter_mm','beam.velocity_jitter_m_per_s',
      'beam.collision_cross_section_m2','beam.gas_velocity_init_mode',
      'beam.source_temperature_k','beam.source_axial_velocity_m_per_s',
      'beam.source_birth_velocity_gas_csv','beam.source_birth_velocity_z_min_mm',
      'beam.source_birth_velocity_z_max_mm','beam.source_birth_velocity_radius_mm',
      'beam.source_radial_velocity_scale',
      'beam.source_velocity_delta_m_per_s',
      'field_bake.simion_dc_path','field_bake.simion_rf_path','field_bake.gas_field_mode','field_bake.fluent_path',
      'field_bake.z_max_mm','field_bake.r_max_mm','field_bake.dz_mm','field_bake.dr_mm',
      'field_bake.pa_grids_per_mm','field_bake.offset_z_simion_mm','field_bake.offset_z_fluent_mm','field_bake.background_pressure_pa',
      'field_bake.background_temperature_k','field_bake.capillary_exit_z_mm',
      'field_bake.fluent_capillary_total_length_mm','field_bake.fluent_capillary_radius_mm',
      'field_bake.fluent_capillary_external_start_z_mm','field_bake.output_npy',
      'runtime.backend','runtime.total_time_s','runtime.macro_dt_s','runtime.random_seed','runtime.snapshot_every',
      'runtime.pic_space_charge_scale',__IICT_FIELD_IDS__,
      'runtime.rf_peak_voltage_v','runtime.rf_frequency_hz','runtime.rf_phase_deg',
      'runtime.pic_grid_nr','runtime.pic_grid_nz',
      'runtime.pic_poisson_backend','runtime.pic_poisson_preconditioner','runtime.pic_poisson_warm_start',
      'runtime.pic_amg_mode','runtime.pic_amg_solver','runtime.pic_amg_tolerance',
      'runtime.pic_amg_max_iters','runtime.pic_amg_fallback_backend',
      'runtime.detector_z_mm','runtime.detector_radius_mm','runtime.radial_limit_mm',
      'runtime.capillary_voltage_v','runtime.capillary_prefill_macro_particles',
      'runtime.macro_particles_per_injection','runtime.max_macro_particle_weight',
      'runtime.max_wall_time_s','runtime.max_terminal_event_rows','runtime.collision_batch_size',
      'runtime.langevin_z_start_mm','runtime.langevin_z_end_mm','runtime.langevin_switch_prob',
      'runtime.langevin_max_dt_s','runtime.electrode_mask_z_offset_mm','runtime.electrode_hit_distance_mm',
      'runtime.collision_mode','runtime.terminal_event_mode','runtime.electrode_mask_path',
      'runtime.electrode_mask_cache_path','runtime.fragmentation_mode',
      'output.trajectory_sample_count','output.trajectory_record_every','output.snapshot_plot_max_points',
      'output.terminal_time_bin_ms',
      'output.save_snapshots','output.save_h5','output.save_figures','output.generate_report_after_run'
    ];
    function setNested(obj, path, value) {
      const parts = path.split('.');
      let cur = obj;
      for (let i = 0; i < parts.length - 1; i++) cur = cur[parts[i]] = cur[parts[i]] || {};
      cur[parts[parts.length - 1]] = value;
    }
    function collect() {
      const payload = {};
      const optionalNumbers = new Set([
        'beam.kinetic_energy_ev','beam.source_temperature_k',
        'beam.source_axial_velocity_m_per_s','beam.source_birth_velocity_z_max_mm',
        'beam.source_birth_velocity_radius_mm',
        'runtime.electrode_mask_z_offset_mm',__IICT_OPTIONAL_NUMBER_IDS__
      ]);
      for (const id of ids) {
        const el = document.getElementById(id);
        if (!el) continue;
        let value = el.type === 'checkbox' ? el.checked : el.value;
        if (el.type === 'number') value = optionalNumbers.has(id) && value === '' ? '' : Number(value);
        setNested(payload, id, value);
      }
      return payload;
    }
    let pendingApply = null;
    async function applyCurrentConfig() {
      if (pendingApply) {
        clearTimeout(pendingApply);
        pendingApply = null;
      }
      const response = await fetch('/api/apply-config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(collect())});
      if (!response.ok) console.warn(await response.text());
      await refresh();
    }
    function scheduleApply() {
      if (pendingApply) clearTimeout(pendingApply);
      pendingApply = setTimeout(applyCurrentConfig, 250);
    }
    async function postAction(path) {
      if (path !== '/api/apply-config') {
        await applyCurrentConfig();
      }
      await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(collect())});
      await refresh();
    }
    async function saveProject() { await postAction('/api/save-config'); }
    async function loadProject() {
      const path = prompt('Project JSON path to load');
      if (!path) return;
      await fetch('/api/load-config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({path})});
      await refresh();
    }
    async function newProject() { await fetch('/api/new-session', {method:'POST'}); await refresh(); }
    function showConfig(id) {
      document.getElementById(id).scrollIntoView({behavior:'smooth', block:'start'});
    }
    function showHelp(topic) {
      const dialog = document.getElementById('help-dialog');
      document.getElementById('help-title').textContent = helpContent[topic].title;
      document.getElementById('help-body').textContent = helpContent[topic].body;
      dialog.showModal();
    }
    function applyConfig(config) {
      const flat = {};
      function walk(prefix, value) {
        if (value && typeof value === 'object' && !Array.isArray(value)) {
          for (const [k, v] of Object.entries(value)) walk(prefix ? prefix + '.' + k : k, v);
        } else flat[prefix] = value;
      }
      walk('', config);
      for (const id of ids) {
        const el = document.getElementById(id);
        if (el && flat[id] !== undefined && document.activeElement !== el) {
          if (el.type === 'checkbox') el.checked = Boolean(flat[id]);
          else el.value = flat[id] ?? '';
        }
      }
      syncConditionalPanels();
    }
    function showPanel(id, visible) {
      document.getElementById(id).classList.toggle('hidden', !visible);
    }
    function syncConditionalPanels() {
      const beamGas = document.getElementById('beam.gas_velocity_init_mode').value === 'from-gas-field';
      showPanel('beam-static-fields', !beamGas);
      showPanel('beam-gas-fields', beamGas);
      const importedGas = document.getElementById('field_bake.gas_field_mode').value === 'import';
      showPanel('field-static-gas-fields', !importedGas);
      showPanel('field-import-gas-fields', importedGas);
      const amg = document.getElementById('runtime.pic_poisson_backend').value === 'amg';
      showPanel('pic-amg-fields', amg);
__IICT_SYNC_JS__
      const hybrid = document.getElementById('runtime.collision_mode').value === 'hybrid-langevin';
      showPanel('hybrid-collision-fields', hybrid);
    }
    async function refresh() {
      const response = await fetch('/api/state');
      const state = await response.json();
      applyConfig(state.config);
      document.getElementById('status').textContent = state.status;
      document.getElementById('beam_summary').textContent = state.beam_summary;
      document.getElementById('progress').textContent = state.progress_line;
      document.getElementById('logs').textContent = state.logs.join('\n');
      document.getElementById('warnings').textContent = state.warnings.join('\n');
      document.getElementById('validation').textContent = state.validation.join('\n');
      document.getElementById('run_summary').textContent = state.run_summary.join('\n');
      const terminalBody = document.getElementById('terminal_rows');
      terminalBody.innerHTML = state.terminal_bins.map(row => {
        const active = row.active_macro == null ? '-' : Number(row.active_macro).toFixed(0);
        const other = Number(row.loss_macro) - Number(row.electrode_hit_macro);
        return `<tr><td>${Number(row.start_ms).toFixed(3)}-${Number(row.end_ms).toFixed(3)}</td><td>${active}</td><td>${Number(row.z_exit_current_na).toPrecision(6)}</td><td>${Number(row.z_exit_real_ions).toPrecision(6)}</td><td>${Number(row.z_exit_macro).toFixed(0)}</td><td>${Number(row.electrode_hit_macro).toFixed(0)}</td><td>${other.toFixed(0)}</td></tr>`;
      }).join('');
      const canvas = document.getElementById('canvas');
      if (state.latest_image_url) {
        canvas.innerHTML = '<img alt="current figure" src="' + state.latest_image_url + '&t=' + Date.now() + '">';
      } else {
        canvas.innerHTML = '<span class="empty">No figure yet.</span>';
      }
    }
    for (const id of ids) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.addEventListener('change', () => {
        syncConditionalPanels();
        scheduleApply();
      });
    }
    setInterval(refresh, 1000);
    syncConditionalPanels();
    refresh();
  </script>
</body>
</html>
"""

_HELP_CONTENT = {
    "parameters": {
        "title": "Parameter Reference",
        "body": parameter_reference_text(),
    },
    "workflow": {
        "title": "Simulation Workflow",
        "body": simulation_workflow_text(),
    },
    "about": {"title": "About MuCITE", "body": about_text()},
}
INDEX_HTML = inject_iict_controls(INDEX_HTML)
INDEX_HTML = INDEX_HTML.replace(
    "__HELP_CONTENT__",
    json.dumps(_HELP_CONTENT, ensure_ascii=False).replace("</", "<\\/"),
)

WebGuiHandler.index_html = INDEX_HTML


def launch_web_window(host: str = "127.0.0.1", port: int = 0) -> None:
    """Launch the browser-based GUI and block until interrupted."""

    state = WebGuiState()

    class Handler(WebGuiHandler):
        pass

    Handler.state = state
    Handler.index_html = INDEX_HTML
    server = ThreadingHTTPServer((host, int(port)), Handler)
    actual_host, actual_port = server.server_address
    url = f"http://{actual_host}:{actual_port}/"
    print(f"MuCITE web GUI: {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.shutdown()
        server.server_close()
