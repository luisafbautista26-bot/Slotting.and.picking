"""Módulo mínimo para picking usado por la app Streamlit.

Implementación reducida que cumple las invariantes requeridas y es segura de
importar. Exporta `nsga2_picking_streamlit` y las constantes globales.
"""

from typing import List, Dict, Any, Iterable
import numpy as np
import random

DEFAULT_VM_PER_SLOT = 3
PROHIBITED_SLOT_INDICES = list(range(4))


def _ensure_vu_map(VU: Iterable) -> Dict[int, float]:
    if VU is None:
        return {}
    VU_arr = np.asarray(list(VU))
    return {i + 1: float(VU_arr[i]) for i in range(VU_arr.shape[0])}


def _evaluate(slot_assignment, Sr, D_racks_arr, VU):
    NUM_SLOTS = len(slot_assignment)
    Vm = np.full(NUM_SLOTS, DEFAULT_VM_PER_SLOT)
    slot_to_rack = list(np.argmax(Sr, axis=1)) if Sr is not None else [0] * NUM_SLOTS
    discharge_racks = set(D_racks_arr) if D_racks_arr is not None else set()
    distancia = 0.0
    penalizado = False
    for i, s in enumerate(slot_assignment):
        if s in PROHIBITED_SLOT_INDICES:
            penalizado = True
            distancia += 100.0
            continue
        rack = slot_to_rack[i] if i < len(slot_to_rack) else 0
        if rack in discharge_racks:
            penalizado = True
            distancia += 50.0
            continue
        distancia += float(i)
    return float(distancia), bool(penalizado)


def nsga2_picking_streamlit(slot_assignments: List[Iterable[int]], D, VU, Sr, D_racks_arr,
                            start_rack=0, box_volume_max=1.0, prohibited_slots=None, seed=None):
    """Evalúa cada solución de slotting y devuelve resultados simples.

    Devuelve lista de dicts con claves: distancia_total, penalizado, discharge_racks
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    prohibited = list(prohibited_slots) if prohibited_slots is not None else PROHIBITED_SLOT_INDICES
    slot_to_rack = list(np.argmax(Sr, axis=1)) if Sr is not None else []
    DISCHARGE_RACKS = sorted(list({slot_to_rack[s] for s in prohibited if 0 <= s < len(slot_to_rack)})) if slot_to_rack else []

    results = []
    for sa in slot_assignments:
        dist, penal = _evaluate(sa, Sr, D_racks_arr, VU)
        results.append({'distancia_total': dist, 'penalizado': penal, 'discharge_racks': DISCHARGE_RACKS})
    return results
