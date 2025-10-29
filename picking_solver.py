"""Picking solver (limpio y compatible con streamlit_app).

Este módulo expone:
- DEFAULT_VM_PER_SLOT: volumen por defecto por slot (3)
- PROHIBITED_SLOTS: índices de slots prohibidos por defecto {0,1,2,3}
- DISCHARGE_RACKS: valor por defecto que puede ser sobreescrito
- nsga2_picking_streamlit(...): wrapper usado por la UI para evaluar
  todas las soluciones de slotting y devolver evaluaciones y Pareto.

La intención es ser robusto frente a entradas: VU puede ser dict o lista;
D_racks puede contener NaN (se sustituyen por 1e6).
"""

from typing import List, Dict, Any, Tuple
import copy
import numpy as np

# Defaults
DEFAULT_VM_PER_SLOT = 3
PROHIBITED_SLOTS = {0, 1, 2, 3}
DISCHARGE_RACKS = [0, 1, 2, 3, 4]


# ----------------------------
# Utilitarios
# ----------------------------
def _build_vu_map(VU_input) -> Dict[int, float]:
    if VU_input is None:
        return {}
    if isinstance(VU_input, dict):
        return {int(k): float(v) for k, v in VU_input.items()}
    arr = list(VU_input)
    return {i + 1: float(arr[i]) for i in range(len(arr))}

"""Picking solver (limpio y compatible con streamlit_app).

Este módulo expone:
- DEFAULT_VM_PER_SLOT: volumen por defecto por slot (3)
- PROHIBITED_SLOTS: índices de slots prohibidos por defecto {0,1,2,3}
- DISCHARGE_RACKS: valor por defecto que puede ser sobreescrito
- nsga2_picking_streamlit(...): wrapper usado por la UI para evaluar
  todas las soluciones de slotting y devolver evaluaciones y Pareto.

La intención es ser robusto frente a entradas: VU puede ser dict o lista;
D_racks puede contener NaN (se sustituyen por 1e6).
"""

from typing import List, Dict, Any, Tuple
import numpy as np

# Defaults
DEFAULT_VM_PER_SLOT = 3
PROHIBITED_SLOTS = {0, 1, 2, 3}
DISCHARGE_RACKS = [0, 1, 2, 3, 4]


# ----------------------------
# Utilitarios
# ----------------------------
def _build_vu_map(VU_input) -> Dict[int, float]:
    if VU_input is None:
        return {}
    if isinstance(VU_input, dict):
        return {int(k): float(v) for k, v in VU_input.items()}
    arr = list(VU_input)
    # if arr contains pairs [sku, vol] detect that
    if arr and hasattr(arr[0], '__len__') and len(arr[0]) >= 2:
        try:
            return {int(row[0]): float(row[1]) for row in arr}
        except Exception:
            pass
    return {i + 1: float(arr[i]) for i in range(len(arr))}


def _clean_D_racks(D_racks: Any) -> np.ndarray:
    Dr = np.array(D_racks, dtype=float)
    Dr = np.nan_to_num(Dr, nan=1e6)
    return Dr


def _slot_to_rack_from_Sr(Sr: Any) -> List[int]:
    Sr_arr = np.array(Sr)
    if Sr_arr.ndim == 1:
        return [0] * len(Sr_arr)
    return np.argmax(Sr_arr, axis=1).tolist()


def hv_2d_min(F: np.ndarray, ref: Tuple[float, float]) -> float:
    # Hypervolume for 2D minimization: area between ref and points
    if F is None or len(F) == 0:
        return 0.0
    pts = np.array(sorted(list(F), key=lambda x: x[0]))
    hv = 0.0
    for x, y in pts:
        width = max(0.0, ref[0] - float(x))
        height = max(0.0, ref[1] - float(y))
        hv += width * height
    return float(hv)


# ----------------------------
# Evaluación y heurísticas básicas
# ----------------------------
def capacity_of_slot(slot_idx: int, Vm_array, sku_id: int, VU_map: Dict[int, float]) -> int:
    if sku_id == 0:
        return 0
    if hasattr(Vm_array, '__len__'):
        Vm_slot = float(Vm_array[slot_idx])
    else:
        Vm_slot = float(Vm_array)
    unit_vol = VU_map.get(int(sku_id), 0.0)
    if unit_vol <= 0:
        return 0
    return int(Vm_slot // unit_vol)


def _evaluate_slot_assignment(slot_row: List[int], D: np.ndarray, VU_map: Dict[int, float],
                              slot_to_rack: List[int], D_racks: np.ndarray,
                              Vm_per_slot=DEFAULT_VM_PER_SLOT, prohibited_slots=set(), start_rack=0):
    """Evalúa una asignación de slots (slot_row) sobre todos los pedidos D.

    Devuelve: (total_distance, sku_distance, penalized_flag, rutas_list, augmented_genome)
    """
    n_orders = int(D.shape[0])
    total_distance = 0.0
    sku_distance = 0.0
    penalized = False
    rutas_all = []

    # per-slot volume
    if hasattr(Vm_per_slot, '__len__'):
        Vm_local_base = np.array(Vm_per_slot, dtype=float)
    else:
        Vm_local_base = np.array([Vm_per_slot] * len(slot_row), dtype=float)

    for oi in range(n_orders):
        order = D[oi]
        Vm_local = Vm_local_base.copy()
        racks_visited = []
        remaining = {i + 1: int(order[i]) for i in range(order.size) if order[i] > 0}
        for sku_id, qty in remaining.items():
            need = qty
            for sidx, ssku in enumerate(slot_row):
                if int(ssku) != int(sku_id):
                    continue
                if sidx in prohibited_slots:
                    continue
                unit_vol = VU_map.get(int(sku_id), 0.0)
                if unit_vol <= 0:
                    continue
                cap = int(Vm_local[sidx] // unit_vol)
                if cap <= 0:
                    continue
                take = min(need, cap)
                if take > 0:
                    r = int(slot_to_rack[sidx])
                    if r not in racks_visited:
                        racks_visited.append(r)
                    Vm_local[sidx] -= take * unit_vol
                    need -= take
                    if need <= 0:
                        break
            if need > 0:
                penalized = True
                total_distance += 1e6 * need

        # sumar distancia desde start hacia cada rack visitado (aprox.)
        for r in racks_visited:
            total_distance += float(D_racks[int(start_rack), int(r)])
        sku_distance += sum(float(D_racks[int(start_rack), int(r)]) for r in racks_visited)
        rutas_all.append([int(start_rack)] + [int(r) for r in racks_visited])

    augmented = None
    return float(total_distance), float(sku_distance), bool(penalized), rutas_all, augmented


def _fast_non_dominated_indices(objs: List[Tuple[float, float]]) -> List[int]:
    n = len(objs)
    S = [set() for _ in range(n)]
    n_dom = [0] * n
    front0 = []
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            a = objs[p]
            b = objs[q]
            if (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1]):
                S[p].add(q)
            elif (b[0] <= a[0] and b[1] <= a[1]) and (b[0] < a[0] or b[1] < a[1]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            front0.append(p)
    return front0


def nsga2_picking_streamlit(slot_assignments: List[List[int]], D, VU, Sr, D_racks,
                            pop_size: int = 20, n_gen: int = 10,
                            px: float = 0.8, pm: float = 0.2,
                            box_volume_max: float = 1.0, start_rack: int = 0,
                            prohibited_slots: List[int] = None, seed: int = 42,
                            verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluación compacta compatible con `streamlit_app.py`.

    Para simplicidad esta implementación evalúa cada solución de slotting
    como un individuo (genoma) y retorna el frente de Pareto entre esas
    soluciones. Mantiene la estructura esperada por la UI.
    """
    if prohibited_slots is None:
        prohibited = set(PROHIBITED_SLOTS)
    else:
        prohibited = set(prohibited_slots)

    VU_map = _build_vu_map(VU)
    Dr = _clean_D_racks(D_racks)
    slot_to_rack = _slot_to_rack_from_Sr(Sr)
    D_arr = np.array(D, dtype=float)

    evaluations = []
    objectives = []
    for cid, srow in enumerate(slot_assignments):
        d, sku_dist, penalized, rutas, aug = _evaluate_slot_assignment(
            slot_row=srow, D=D_arr, VU_map=VU_map, slot_to_rack=slot_to_rack,
            D_racks=Dr, Vm_per_slot=DEFAULT_VM_PER_SLOT, prohibited_slots=prohibited,
            start_rack=start_rack)
        # compute augmented genome for UI: build a best-effort genome from rutas
        augmented = None
        try:
            # best-effort genome: [cluster_idx, 0, racks_of_order1..., 0, racks_of_order2..., 0, ...]
            genome_guess = [cid, 0]
            for route in rutas:
                # route is [start_rack, r1, r2, ...]
                middle = [int(r) for r in route[1:]]
                genome_guess.extend(middle)
                genome_guess.append(0)

            # expose D_racks and VU_map for helper
            globals()['_CURRENT_D_RACKS'] = Dr
            globals()['_CURRENT_VU_MAP'] = VU_map
            augmented = insert_discharge_points_and_boxes(genome_guess, slot_assignments, slot_to_rack,
                                                          box_volume_max, start_rack, D_arr)
        except Exception:
            augmented = aug
        finally:
            globals().pop('_CURRENT_D_RACKS', None)
            globals().pop('_CURRENT_VU_MAP', None)

        evaluations.append((d, sku_dist, penalized, rutas, augmented))
        objectives.append((d, sku_dist))

    # calcular frente de Pareto entre las soluciones
    pareto_idx = _fast_non_dominated_indices(objectives)
    pareto_front = [(i, float(objectives[i][0]), float(objectives[i][1])) for i in pareto_idx]

    # construir dict de salida compatible con streamlit_app expectations
    population_eval_triples = evaluations

    if population_eval_triples:
        best0 = population_eval_triples[pareto_idx[0] if pareto_idx else 0]
        distancia_total = float(best0[0])
        sku_distancia = float(best0[1])
        penalizado = bool(best0[2])
        rutas_best = best0[3]
        augmented_best = best0[4]
    else:
        distancia_total = None
        sku_distancia = None
        penalizado = True
        rutas_best = []
        augmented_best = None

    result = {
        'pareto_front': pareto_front,
        'population_eval_triples': population_eval_triples,
        'distancia_total': distancia_total,
        'sku_distancia': sku_distancia,
        'penalizado': penalizado,
        'rutas_best': rutas_best,
        'augmented_best': augmented_best,
    }
    return result



def _split_genome_orders(genome: List[int], start_rack: int = 0) -> Tuple[int, List[List[int]]]:
    """Parsea un genome que puede opcionalmente empezar con cluster_idx seguido de 0.

    Retorna (cluster_idx_or_None, list_of_order_blocks).
    Cada block es una lista de racks visitados (sin separador 0).
    """
    if genome is None:
        return None, []
    g = [int(x) for x in genome]
    cluster_idx = None
    idx = 0
    if len(g) >= 2 and g[1] == 0:
        cluster_idx = int(g[0])
        idx = 2
    blocks = []
    cur = []
    for val in g[idx:]:
        if val == 0:
            blocks.append(cur)
            cur = []
        else:
            cur.append(int(val))
    if cur:
        blocks.append(cur)
    return cluster_idx, blocks


def _ensure_discharge_in_blocks(blocks: List[List[int]], discharge_racks: List[int], D_racks: np.ndarray, start_rack: int = 0) -> List[List[int]]:
    """Asegura que cada block termina con un rack de descarga; si no, inserta el más cercano."""
    out = []
    if not discharge_racks:
        discharge_racks = DISCHARGE_RACKS
    # sanitize discharge_racks against D_racks shape
    valid_dr = []
    try:
        max_r = int(D_racks.shape[0])
    except Exception:
        max_r = None
    if max_r is None:
        valid_dr = list(discharge_racks)
    else:
        for dr in discharge_racks:
            if 0 <= int(dr) < max_r:
                valid_dr.append(int(dr))
    if not valid_dr:
        # fallback to 0 if nothing valid
        valid_dr = [0]

    for blk in blocks:
        if not blk:
            # si no hay visitas, insertar el discharge más cercano al start
            try:
                nearest = int(min(valid_dr, key=lambda dr: float(D_racks[start_rack, int(dr)])))
            except Exception:
                nearest = int(valid_dr[0])
            out.append([nearest])
            continue
        last = int(blk[-1])
        if last in discharge_racks:
            out.append(blk)
        else:
            # elegir discharge más cercano al último rack visitado
            try:
                nearest = int(min(valid_dr, key=lambda dr: float(D_racks[last, int(dr)])))
            except Exception:
                nearest = int(valid_dr[0])
            out.append(blk + [nearest])
    return out


def normalize_genome(genome: List[int], D_racks: Any, start_rack: int = 0, discharge_racks: List[int] = None) -> List[int]:
    """Normaliza un genome para que cada pedido termine con un rack de descarga seguido de 0.

    Mantiene el prefijo cluster_idx si existe.
    """
    if discharge_racks is None:
        discharge_racks = DISCHARGE_RACKS
    cluster_idx, blocks = _split_genome_orders(genome, start_rack=start_rack)
    Dr = _clean_D_racks(D_racks)
    blocks = _ensure_discharge_in_blocks(blocks, discharge_racks, Dr, start_rack=start_rack)
    out = []
    if cluster_idx is not None:
        out.extend([int(cluster_idx), 0])
    for blk in blocks:
        out.extend([int(x) for x in blk])
        out.append(0)
    if not out:
        # minimal genome: just a separator
        out = [0]
    if out[-1] != 0:
        out.append(0)
    return out


def evaluate_individual(genome: List[int], slot_assignments: List[List[int]], slot_to_rack: List[int], VU_map: Dict[int, float],
                        Vm_per_slot, D_racks: Any, orders: np.ndarray, box_volume_max: float = 1.0,
                        start_rack: int = 0, prohibited_slots: set = None) -> Tuple[float, float, bool, List[List[int]], Any]:
    """Evalúa un individuo descrito por `genome`.

    El genome puede contener un prefijo cluster_idx (p.ej. [cid,0,...]) o no.
    El formato de bloques entre ceros representa las visitas por pedido. Esta función
    asegura que antes del separador 0 exista un rack de descarga y simula la recogida
    de SKUs desde los slots del cluster seleccionado.
    """
    if prohibited_slots is None:
        prohibited_slots = set(PROHIBITED_SLOTS)
    Dr = _clean_D_racks(D_racks)
    cluster_idx, blocks = _split_genome_orders(genome, start_rack=start_rack)
    if cluster_idx is None:
        cluster_idx = 0
    # seguridad sobre índices
    if cluster_idx < 0 or cluster_idx >= len(slot_assignments):
        cluster_idx = 0
    slot_row = slot_assignments[int(cluster_idx)]

    total_distance = 0.0
    sku_distance = 0.0
    penalized = False
    rutas = []

    # asegurar discharge racks en bloques
    blocks = _ensure_discharge_in_blocks(blocks, DISCHARGE_RACKS, Dr, start_rack=start_rack)

    num_orders = int(orders.shape[0]) if orders is not None else len(blocks)
    for oi in range(min(num_orders, len(blocks))):
        blk = blocks[oi]
        # construir ruta: start -> blk... -> start
        route = [int(start_rack)] + [int(x) for x in blk] + [int(start_rack)]
        rutas.append(route)

        # simular picks
        remaining = {i + 1: int(orders[oi][i]) for i in range(orders.shape[1]) if orders[oi][i] > 0}
        cur = int(start_rack)
        for nxt in route[1:]:
            # sumar distancia de cur->nxt
            try:
                total_distance += float(Dr[cur, int(nxt)])
            except Exception:
                total_distance += 0.0
            # intentar recoger en ese rack
            for slot_idx, sku in enumerate(slot_row):
                if int(slot_to_rack[slot_idx]) != int(nxt):
                    continue
                sku_id = int(sku)
                if sku_id == 0:
                    continue
                if sku_id not in remaining:
                    continue
                cap = capacity_of_slot(slot_idx, Vm_per_slot, sku_id, VU_map)
                if cap <= 0:
                    continue
                take = min(cap, remaining[sku_id])
                remaining[sku_id] -= take
                if remaining[sku_id] <= 0:
                    del remaining[sku_id]
            cur = int(nxt)

        if remaining:
            penalized = True
            total_distance += 1e6 * sum(remaining.values())

        # sku_distance: suma de distancias a los racks donde se recogió algo (proxy)
        # aquí simplificamos: suma distancias desde start a cada rack visited (sin repetir)
        visited = set(blk)
        sku_distance += sum(float(Dr[start_rack, int(r)]) for r in visited)

    augmented = None
    # build augmented genome with explicit discharge points (and potential box metadata)
    try:
        # expose D_racks and VU_map for the helper
        globals()['_CURRENT_D_RACKS'] = Dr
        globals()['_CURRENT_VU_MAP'] = VU_map
        augmented = insert_discharge_points_and_boxes(genome, slot_assignments, slot_to_rack,
                                                      box_volume_max, start_rack, orders)
    except Exception:
        augmented = None
    finally:
        globals().pop('_CURRENT_D_RACKS', None)
        globals().pop('_CURRENT_VU_MAP', None)

    return float(total_distance), float(sku_distance), bool(penalized), rutas, augmented


def insert_discharge_points_and_boxes(genome: List[int], slot_assignments: List[List[int]],
                                      slot_to_rack_local: List[int], box_volume_max: float,
                                      start_rack: int, orders: np.ndarray,
                                      discharge_racks: List[int] = None) -> List[int]:
    """Inserta puntos de descarga en el genome y devuelve un genome aumentado.

    Esta implementación ligera garantiza que cada bloque (separado por 0) termine en
    un rack de descarga. No hace un cálculo exhaustivo de cajas (boxes) aquí — solo
    asegura los puntos de descarga que la UI necesita ver. La firma coincide con la
    llamada solicitada por el UI: insert_discharge_points_and_boxes(genome, slot_assignments,
    slot_to_rack_local, box_volume_max, start_rack, orders)
    """
    # This implementation follows the structure requested by the user.
    # It expects to read D_racks and VU_map from temporary globals set by the caller.
    D_racks = globals().get('_CURRENT_D_RACKS', None)
    VU_map = globals().get('_CURRENT_VU_MAP', {})

    if discharge_racks is None:
        discharge_racks = DISCHARGE_RACKS

    # defensive
    if genome is None or len(genome) == 0:
        return [0]

    try:
        cluster_idx = int(genome[0])
    except Exception:
        cluster_idx = 0

    # guard cluster idx
    if cluster_idx < 0 or cluster_idx >= len(slot_assignments):
        cluster_idx = 0

    slot_assignment_row = slot_assignments[int(cluster_idx)]
    new_genome = [int(cluster_idx), 0]
    i = 2
    order_idx = 0
    current_rack = int(start_rack)
    box_vol = 0.0
    try:
        orders_arr = np.array(orders)
    except Exception:
        orders_arr = np.array([])
    order_demands = [dict((int(sku) + 1, int(qty)) for sku, qty in enumerate(order) if qty > 0) for order in orders_arr]
    num_orders = len(order_demands)

    if D_racks is None:
        D_racks = np.zeros((1, 1), dtype=float)
    else:
        D_racks = _clean_D_racks(D_racks)

    while i < len(genome):
        try:
            rack = int(genome[i])
        except Exception:
            i += 1
            continue
        if rack == 0:
            # before closing order, insert nearest discharge if last wasn't discharge
            if new_genome[-1] not in discharge_racks:
                try:
                    nearest_discharge = min(discharge_racks, key=lambda dp: D_racks[current_rack, dp])
                    nearest_discharge = int(nearest_discharge)
                except Exception:
                    nearest_discharge = int(discharge_racks[0]) if discharge_racks else 0
                new_genome.append(nearest_discharge)
                current_rack = nearest_discharge
                box_vol = 0.0
            new_genome.append(0)
            order_idx += 1
            current_rack = int(start_rack)
            box_vol = 0.0
            i += 1
            continue

        new_genome.append(rack)
        current_rack = rack

        if order_idx < num_orders:
            demand = order_demands[order_idx]
            # simulate picks at this rack: scan slots that belong to this rack
            for slot_idx, sku in enumerate(slot_assignment_row):
                try:
                    if slot_to_rack_local[slot_idx] != rack:
                        continue
                except Exception:
                    continue
                sku_id = int(sku)
                if sku_id == 0:
                    continue
                if sku_id not in demand or demand[sku_id] <= 0:
                    continue
                qty_to_pick = demand[sku_id]
                unit_vol = float(VU_map.get(sku_id, 0.0))
                for _ in range(qty_to_pick):
                    if box_vol + unit_vol > box_volume_max:
                        try:
                            nearest_discharge = min(discharge_racks, key=lambda dp: D_racks[current_rack, dp])
                            nearest_discharge = int(nearest_discharge)
                        except Exception:
                            nearest_discharge = int(discharge_racks[0]) if discharge_racks else 0
                        new_genome.append(nearest_discharge)
                        current_rack = nearest_discharge
                        box_vol = 0.0
                    box_vol += unit_vol
                demand[sku_id] = 0
        i += 1

    # clean duplicates
    filtered = [new_genome[0], new_genome[1]]
    for j in range(2, len(new_genome)):
        if new_genome[j] in discharge_racks and filtered[-1] == new_genome[j]:
            continue
        filtered.append(new_genome[j])
    return filtered


__all__ = ['nsga2_picking_streamlit', 'DEFAULT_VM_PER_SLOT', 'PROHIBITED_SLOTS', 'DISCHARGE_RACKS', 'normalize_genome', 'evaluate_individual', 'insert_discharge_points_and_boxes']
