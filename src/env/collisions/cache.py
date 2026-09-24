"""IonSPA thermodynamic-model cache."""

from __future__ import annotations

from typing import Any

from scipy.constants import Avogadro

from ...config import IonTemplate
from .models import ApproximateIonModel


class IonModelCacheMixin:
    """Cache one IonSPA model for each immutable template signature."""

    @staticmethod
    def _ion_model_cache_key(template: IonTemplate) -> tuple[Any, ...]:
        return (
            template.name,
            float(template.mass_kg),
            int(template.charge_state),
            float(template.collision_cross_section_m2),
            int(template.num_atoms),
            float(template.initial_internal_temperature_k),
            template.heat_capacity_profile,
            float(template.delta_h_kj_per_mol),
            float(template.delta_s_j_per_mol_k),
        )

    def _build_ion_model(self, template: IonTemplate) -> Any:
        cache_key = self._ion_model_cache_key(template)
        cached = self._ion_model_cache.get(cache_key)
        if cached is not None:
            return cached
        if self.ionspa is None:
            ion_model = ApproximateIonModel(template)
            self._ion_model_cache[cache_key] = ion_model
            return ion_model
        ion_dict = dict(
            name=template.name,
            mass=template.mass_kg * Avogadro,
            charge=template.charge_state,
            CCS=template.collision_cross_section_m2 * 1.0e18,
            num_atoms=template.num_atoms,
            T0=template.initial_internal_temperature_k,
            hcprofname=template.heat_capacity_profile,
            refdH=template.delta_h_kj_per_mol,
            refdS=template.delta_s_j_per_mol_k,
        )
        ion_model = self.ionspa.ionclass(ion_dict)
        self._ion_model_cache[cache_key] = ion_model
        return ion_model


__all__ = ["IonModelCacheMixin"]
