"""Browser-form fragments for collision-physics backend configuration."""

from __future__ import annotations

import json


IICT_FIELD_IDS = (
    "runtime.collision_physics_backend",
    "runtime.ionspa_backend",
    "runtime.iict_parameter_config_path",
    "runtime.iict_heat_capacity_model",
    "runtime.iict_num_atoms",
    "runtime.iict_constant_cv_j_per_k_per_ion",
    "runtime.iict_heat_capacity_csv_path",
    "runtime.iict_pseudoatom_model",
    "runtime.iict_pseudoatom_mass_da",
    "runtime.iict_pseudoatom_csv_path",
    "runtime.iict_pseudoatom_min_mass_da",
    "runtime.iict_pseudoatom_max_mass_da",
    "runtime.iict_fragmentation_model",
    "runtime.iict_delta_h_kj_per_mol",
    "runtime.iict_delta_s_j_per_mol_k",
)

IICT_OPTIONAL_NUMBER_IDS = (
    "runtime.iict_num_atoms",
    "runtime.iict_constant_cv_j_per_k_per_ion",
    "runtime.iict_pseudoatom_mass_da",
    "runtime.iict_pseudoatom_min_mass_da",
    "runtime.iict_pseudoatom_max_mass_da",
    "runtime.iict_delta_h_kj_per_mol",
    "runtime.iict_delta_s_j_per_mol_k",
)

IICT_PANEL_HTML = r"""
        <div class="subgroup">
          <label>Single-collision physics backend</label>
          <select id="runtime.collision_physics_backend"><option>ionspa</option><option>iict-lite</option></select>
          <div id="ionspa-physics-fields">
            <label>IonSPA provider</label>
            <select id="runtime.ionspa_backend"><option>local</option><option>approximate</option></select>
            <div class="statusline">Optional compatibility adapter; use only with a provider/source you are legally permitted to use.</div>
          </div>
          <div id="iict-physics-fields">
            <label>iict-lite parameter JSON (schema v1, optional)</label>
            <input id="runtime.iict_parameter_config_path">
            <div class="statusline">Blank override fields come from JSON. Non-blank GUI values override JSON and are reported as simulation_config sources.</div>
            <div class="two">
              <div><label>Heat model override</label><select id="runtime.iict_heat_capacity_model"><option value="">From JSON</option><option>classical</option><option>constant_cv</option><option>tabulated</option></select></div>
              <div><label>Atom count (blank = Beam)</label><input id="runtime.iict_num_atoms" type="number"></div>
              <div><label>Constant Cv [J/K/ion]</label><input id="runtime.iict_constant_cv_j_per_k_per_ion" type="number" step="any"></div>
              <div><label>Heat-capacity CSV</label><input id="runtime.iict_heat_capacity_csv_path"></div>
              <div><label>Pseudo-atom model override</label><select id="runtime.iict_pseudoatom_model"><option value="">From JSON</option><option>constant</option><option>tabulated</option></select></div>
              <div><label>Pseudo-atom mass [Da]</label><input id="runtime.iict_pseudoatom_mass_da" type="number" step="any"></div>
              <div><label>Pseudo-atom CSV</label><input id="runtime.iict_pseudoatom_csv_path"></div>
              <div><label>Minimum mass [Da]</label><input id="runtime.iict_pseudoatom_min_mass_da" type="number" step="any"></div>
              <div><label>Maximum mass [Da]</label><input id="runtime.iict_pseudoatom_max_mass_da" type="number" step="any"></div>
              <div><label>Rate model override</label><select id="runtime.iict_fragmentation_model"><option value="">From JSON</option><option>none</option><option>eyring</option></select></div>
              <div><label>Eyring delta H [kJ/mol]</label><input id="runtime.iict_delta_h_kj_per_mol" type="number" step="any"></div>
              <div><label>Eyring delta S [J/mol/K]</label><input id="runtime.iict_delta_s_j_per_mol_k" type="number" step="any"></div>
            </div>
            <div class="statusline">iict-lite is an independent paper-driven approximation (DOI 10.1016/j.ijms.2024.117290), not an IonSPA-equivalence claim.</div>
          </div>
        </div>
"""

IICT_SYNC_JS = """
      const iictLite = document.getElementById('runtime.collision_physics_backend').value === 'iict-lite';
      showPanel('ionspa-physics-fields', !iictLite);
      showPanel('iict-physics-fields', iictLite);
"""


def _js_items(values: tuple[str, ...]) -> str:
    return ",".join(json.dumps(value) for value in values)


def inject_iict_controls(index_html: str) -> str:
    """Replace stable browser-template placeholders with IICT controls."""

    replacements = {
        "__IICT_PANEL_HTML__": IICT_PANEL_HTML,
        "__IICT_FIELD_IDS__": _js_items(IICT_FIELD_IDS),
        "__IICT_OPTIONAL_NUMBER_IDS__": _js_items(IICT_OPTIONAL_NUMBER_IDS),
        "__IICT_SYNC_JS__": IICT_SYNC_JS,
    }
    rendered = index_html
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


__all__ = [
    "IICT_FIELD_IDS",
    "IICT_OPTIONAL_NUMBER_IDS",
    "IICT_PANEL_HTML",
    "inject_iict_controls",
]
