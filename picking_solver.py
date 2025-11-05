# picking_solver.py
# Corregido para devolver y preservar correctamente el "genoma aumentado" (augmented)
# y asegurar que los puntos de descarga se inserten antes de cerrar cada pedido.
import numpy as np
import random
import copy
import matplotlib.pyplot as plt

# Public configuration variables expected by streamlit_app
DEFAULT_VM_PER_SLOT = 3
DISCHARGE_RACKS = [0, 1, 2, 3]
DEFAULT_DISCHARGE_RACKS = DISCHARGE_RACKS

# ----------------------------
# --- PREPROCESSING & HELPERS
# ----------------------------

def _build_vu_array(VU, NUM_SKUS):
    """Return an array-like VU_array indexed 0..NUM_SKUS-1 from VU input.
    VU may be a dict with 1-based keys or an array/list already.
    """
    if VU is None:
        return np.zeros(NUM_SKUS, dtype=float)
    # if dict with 1-based keys
    if isinstance(VU, dict):
        arr = np.zeros(NUM_SKUS, dtype=float)
        for k, v in VU.items():
            try:
                idx = int(k) - 1
                if 0 <= idx < NUM_SKUS:
                    arr[idx] = float(v)
            except Exception:
                continue
        return arr
    # otherwise try to convert to numpy array
    arr = np.asarray(VU, dtype=float)
    if arr.size >= NUM_SKUS:
        return arr[:NUM_SKUS]
    # pad if smaller
    out = np.zeros(NUM_SKUS, dtype=float)
    out[:arr.size] = arr
    return out

# ----------------------------
# --- HV utilities (2D minimization)
# ----------------------------
def is_dominated(p, q):
    return (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1])

def filter_nondominated(points):
    pts = np.array(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 2)
    n = pts.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if is_dominated(pts[i], pts[j]):
                keep[i] = False
                break
            if is_dominated(pts[j], pts[i]):
                keep[j] = False
    return pts[keep]

def hv_2d_min(points, ref):
    pts = np.array(points, dtype=float)
    if pts.size == 0:
        return 0.0
    ref = np.array(ref, dtype=float)
    pts = pts[np.all(pts <= ref, axis=1)]
    if pts.size == 0:
        return 0.0
    pts = pts.reshape(-1, 2)
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_f2 = ref[1]
    ref_f1 = ref[0]
    for x1, x2 in pts:
        if x2 < prev_f2:
            width = max(0.0, ref_f1 - x1)
            height = max(0.0, prev_f2 - x2)
            hv += width * height
            prev_f2 = x2
    return hv

def make_ref_point_from_fronts(fronts, factor=1.2, require_nondominated=True):
    valid_fronts = [np.array(f, dtype=float) for f in fronts if len(f) > 0]
    if not valid_fronts:
        raise ValueError("No fronts to build ref_point.")
    all_pts = np.vstack(valid_fronts)
    if require_nondominated:
        all_pts = filter_nondominated(all_pts)
    if all_pts.size == 0:
        raise ValueError("No valid points for ref_point.")
    return factor * np.max(all_pts, axis=0)

# ----------------------------
# --- Picking / genome logic
# ----------------------------
def capacity_of_slot(slot_idx, Vm_array, sku_id, VU_map):
    if sku_id == 0:
        return 0
    Vm_slot = Vm_array[slot_idx] if hasattr(Vm_array, "__len__") else Vm_array
    unit_vol = VU_map.get(sku_id, 0.0)
    if unit_vol <= 0:
        return 0
    return int(Vm_slot // unit_vol)

def route_for_order(order, slot_assignment_row, Vm_array_local, slot_to_rack_local, start_rack, VU_map, D_racks, PROHIBITED_SLOT_INDICES):
    Vm_local = Vm_array_local.copy() if hasattr(Vm_array_local, "__len__") else np.array([Vm_array_local]*len(slot_assignment_row))
    route = [start_rack]
    indices = np.where(order > 0)[0]
    demandas = order[order > 0].astype(int)
    remaining = {int(idx+1): int(q) for idx, q in zip(indices, demandas)}
    racks_needed = set()

    for sku_id, qty in remaining.items():
        need = qty
        for slot_idx, slot_sku in enumerate(slot_assignment_row):
            if int(slot_sku) != sku_id:
                continue
            if slot_idx in PROHIBITED_SLOT_INDICES:
                continue
            unit_vol = VU_map.get(sku_id, 0.0)
            if unit_vol <= 0:
                continue
            cap = int(Vm_local[slot_idx] // unit_vol)
            if cap <= 0:
                continue
            take = min(need, cap)
            Vm_local[slot_idx] -= take * unit_vol
            if take > 0:
                racks_needed.add(slot_to_rack_local[slot_idx])
                need -= take
                if need <= 0:
                    break

    # greedy nearest-first order visiting racks_needed
    racks_needed = list(racks_needed)
    current = start_rack
    route_list = []
    while racks_needed:
        next_rack = min(racks_needed, key=lambda r: D_racks[current, r])
        route_list.append(next_rack)
        current = next_rack
        racks_needed.remove(next_rack)
    route += route_list + [start_rack]
    return route

def build_genome(orders, slot_assignment_row, Vm_array_local, slot_to_rack_local, start_rack, cluster_idx, VU_map, D_racks, PROHIBITED_SLOT_INDICES):
    """
    Construye el genoma asegurando que 0 funciona solo como separador de pedidos.
    Para cada pedido toma sub[1:-1] (quita start_rack al inicio y al final) y añade un 0 separador.
    Si el pedido no visita racks, añade solo el separador 0 (se forzará un punto de descarga después).
    """
    genome = [cluster_idx, 0]
    for order in orders:
        # skip completely empty orders (rows with all zeros) to avoid creating
        # unnecessary 0-separators that later confuse routing/counting.
        if not any((qty > 0) for qty in order):
            # do not add a separator for empty orders; downstream code
            # (insert_discharge_points_and_boxes/evaluate_individual)
            # counts non-empty orders and will produce consistent output.
            continue
        sub = route_for_order(order, slot_assignment_row, Vm_array_local, slot_to_rack_local, start_rack, VU_map, D_racks, PROHIBITED_SLOT_INDICES)
        # sub es [start_rack, r1, r2, ..., start_rack]
        if len(sub) <= 2:
            # pedido sin visitas consideramos que no añadimos racks intermedios;
            # dejamos que insert_discharge_points_and_boxes inserte descarga si procede
            genome += [0]
        else:
            genome += sub[1:-1] + [0]
    return genome

def insert_discharge_points_and_boxes(genome, slot_assignments, slot_to_rack_local,
                                      box_volume_max, start_rack, orders):
    """
    Inserta visitas a racks de descarga cuando corresponda:
     - Si un pedido no visitó racks, inserta nearest_discharge antes de cerrar.
     - Si un pedido termina en un rack que no es descarga, inserta nearest_discharge antes del 0 separador.
     - Si durante la simulación de picks la caja se llena, inserta nearest_discharge en ese punto.
    """
    # This implementation expects module-level globals to be set:
    # D_racks, VU_map, DISCHARGE_RACKS, PROHIBITED_SLOT_INDICES
    global D_racks, VU_map, DISCHARGE_RACKS, PROHIBITED_SLOT_INDICES
    cluster_idx = genome[0]
    slot_assignment_row = slot_assignments[cluster_idx]
    new_genome = [cluster_idx, 0]
    i = 2
    order_idx = 0
    current_rack = start_rack
    box_vol = 0.0
    orders_arr = np.array(orders)
    order_demands = [dict((int(sku)+1, int(qty)) for sku, qty in enumerate(order) if qty > 0) for order in orders_arr]
    # number of non-empty orders (some input files may have empty rows)
    num_orders = sum(1 for d in order_demands if any(q > 0 for q in d.values()))

    # Normalize parameter names: use a local 'discharge_racks' variable (from
    # the incoming parameter). If it's None or empty, fall back to the module
    # default DISCHARGE_RACKS constant.
    discharge_racks = list(DISCHARGE_RACKS) if DISCHARGE_RACKS is not None else list(DEFAULT_DISCHARGE_RACKS)
    discharge_racks = list(DISCHARGE_RACKS) if 'DISCHARGE_RACKS' in globals() and DISCHARGE_RACKS is not None else list(DEFAULT_DISCHARGE_RACKS)
    def append_nearest_discharge(state_list, curr):
        # usar siempre el parámetro local 'discharge_racks' (respeta override)
        if not discharge_racks:
            return curr
        # prefer a discharge rack that is not equal to start_rack and not equal
        # to the last appended rack (avoid 0 -> discharge -> 0 and duplicates)
        candidates = [dp for dp in discharge_racks if dp != start_rack and (not state_list or dp != state_list[-1])]
        if not candidates:
            # fallback: allow any discharge except repeating the last element
            candidates = [dp for dp in discharge_racks if not state_list or dp != state_list[-1]]
            if not candidates:
                return curr
        nearest_discharge = min(candidates, key=lambda dp: D_racks[curr, dp])
        # only append if it's different from the last appended element
        if not state_list or nearest_discharge != state_list[-1]:
            state_list.append(nearest_discharge)
        return nearest_discharge

    while i < len(genome):
        rack = genome[i]
        if rack == 0:
            # Always ensure we add a discharge rack before closing the order (0).
            # If there were no intermediate visits (new_genome[-1] == 0) try to
            # insert a rack that actually contains demanded SKUs; otherwise fall
            # back to inserting the nearest discharge.
            if new_genome[-1] == 0:
                try:
                    # buscar racks que tengan SKUs pedidos para este pedido
                    potential_racks = set()
                    if 0 <= order_idx < len(order_demands):
                        demand_keys = set(k for k, v in order_demands[order_idx].items() if v > 0)
                        for slot_idx, sku in enumerate(slot_assignment_row):
                            try:
                                sku_id = int(sku)
                            except Exception:
                                continue
                            if sku_id == 0:
                                continue
                            if sku_id in demand_keys:
                                potential_racks.add(slot_to_rack_local[slot_idx])
                    # prefer racks that are NOT discharge racks
                    non_discharge = [r for r in potential_racks if r not in discharge_racks]
                    chosen = None
                    if non_discharge:
                        chosen = min(non_discharge, key=lambda r: D_racks[current_rack, r])
                    elif potential_racks:
                        # all potential_racks might be discharge racks; prefer to
                        # avoid creating a 0->discharge->0 route. Try to find
                        # a nearby non-discharge rack that contains *any* SKU
                        # (not necessarily demanded) and visit it instead.
                        chosen = None
                        # first try potential racks (may be discharge)
                        try:
                            chosen = min(potential_racks, key=lambda r: D_racks[current_rack, r])
                        except Exception:
                            chosen = None
                        if chosen is not None and chosen in discharge_racks:
                            # buscar racks no-descarga que contengan cualquier SKU
                            candidate_non_discharge = set()
                            for slot_idx, sku in enumerate(slot_assignment_row):
                                try:
                                    sku_id = int(sku)
                                except Exception:
                                    continue
                                if sku_id == 0:
                                    continue
                                rack_idx = slot_to_rack_local[slot_idx]
                                if rack_idx not in discharge_racks:
                                    candidate_non_discharge.add(rack_idx)
                            if candidate_non_discharge:
                                chosen = min(candidate_non_discharge, key=lambda r: D_racks[current_rack, r])

                    if chosen is not None:
                        # insertar visita al rack elegido antes de cerrar el pedido
                        new_genome.append(chosen)
                        current_rack = chosen
                        # after inserting a discharge-like visit, simulate a return to start
                        # for the purpose of selecting the next nearest rack so we avoid
                        # immediate transitions between different discharge racks.
                        try:
                            current_rack = start_rack
                        except Exception:
                            pass
                        box_vol = 0.0
                    else:
                        # no se encontró rack con SKUs pedidos: insertar descarga como antes
                        nearest = append_nearest_discharge(new_genome, current_rack)
                        current_rack = nearest
                        # same: simulate return-to-start for next selection
                        try:
                            current_rack = start_rack
                        except Exception:
                            pass
                        box_vol = 0.0
                except Exception:
                    # en caso de fallo, comportamiento por defecto: insertar descarga
                    nearest = append_nearest_discharge(new_genome, current_rack)
                    current_rack = nearest
                    box_vol = 0.0

            # Si el último no es punto de descarga, insertar nearest antes de cerrar
            if discharge_racks and new_genome[-1] not in discharge_racks:
                    # compute nearest, append it (append_nearest_discharge will avoid duplicates)
                nearest = append_nearest_discharge(new_genome, current_rack)
                current_rack = nearest
                # simulate return-to-start to avoid immediate discharge->discharge transitions
                try:
                    current_rack = start_rack
                except Exception:
                    pass
                box_vol = 0.0

            # finally, append the 0 separator to close the order
            new_genome.append(0)
            # advance to next non-empty order index: we count separators as closing
            order_idx += 1
            current_rack = start_rack
            box_vol = 0.0
            i += 1
            continue

        # Añadir visita de rack
        new_genome.append(rack)
        current_rack = rack

        if order_idx < num_orders:
            demand = order_demands[order_idx]
            # simular picks en este rack
            for slot_idx, sku in enumerate(slot_assignment_row):
                if slot_to_rack_local[slot_idx] != rack:
                    continue
                sku_id = int(sku)
                if sku_id == 0:
                    continue
                if sku_id not in demand or demand[sku_id] <= 0:
                    continue
                qty_to_pick = demand[sku_id]
                unit_vol = VU_map.get(sku_id, 0.0)
                for _ in range(qty_to_pick):
                    if box_vol + unit_vol > box_volume_max:
                        nearest = append_nearest_discharge(new_genome, current_rack)
                        current_rack = nearest
                        box_vol = 0.0
                    box_vol += unit_vol
                demand[sku_id] = 0
        i += 1

    # Si no se cerró con 0, cerrar con descarga + 0
    if new_genome[-1] != 0:
        nearest = append_nearest_discharge(new_genome, current_rack)
        new_genome.append(0)

    # Limpiar duplicados consecutivos exactos (pero mantener estructura)
    filtered = [new_genome[0], new_genome[1]]
    for j in range(2, len(new_genome)):
        if new_genome[j] == filtered[-1]:
            continue
        filtered.append(new_genome[j])
    # Second-pass cleanup: remove trivial discharge-only subroutes 0 -> d -> 0
    # when d is a discharge rack (they create empty segments). We keep a
    # single 0 separator.
    cleaned = []
    k = 0
    while k < len(filtered):
        if k + 2 < len(filtered) and filtered[k] == 0 and filtered[k+2] == 0 and filtered[k+1] in discharge_racks:
            # skip the middle discharge rack
            cleaned.append(0)
            k += 3
            # collapse possible consecutive zeros
            while k < len(filtered) and filtered[k] == 0:
                k += 1
            continue
        cleaned.append(filtered[k])
        k += 1
    # ensure final genome ends with a 0 separator
    if cleaned and cleaned[-1] != 0:
        cleaned.append(0)

    # Remove consecutive discharge->discharge transitions: keep only the first
    # discharge rack in any run of consecutive discharge racks. These adjacent
    # discharge visits are redundant (no picks happen between them) and create
    # empty subroutes; prefer a single discharge visit instead.
    final = []
    for x in cleaned:
        if final and final[-1] in discharge_racks and x in discharge_racks:
            # skip the later discharge rack
            continue
        final.append(x)

    # ensure final genome ends with a 0 separator (again, after modifications)
    if final and final[-1] != 0:
        final.append(0)
    return final

def most_demanded_sku_distance(genome, slot_assignments, slot_to_rack_local, orders, start_rack=0, top_k=5, D_racks=None):
    cluster_idx = genome[0]
    slot_assignment_row = slot_assignments[cluster_idx]
    sku_demands = np.sum(orders, axis=0)
    nonzero = np.where(sku_demands > 0)[0]
    if nonzero.size == 0:
        return 0.0
    top = nonzero[np.argsort(sku_demands[nonzero])[::-1][:top_k]]
    total = 0.0
    for sku_idx in top:
        sku_id = int(sku_idx) + 1
        slots = [i for i, s in enumerate(slot_assignment_row) if int(s) == sku_id]
        for slot in slots:
            rack = slot_to_rack_local[slot]
            total += D_racks[start_rack, rack]
    return total

def evaluate_individual(genome, slot_assignments, slot_to_rack_local, box_volume_max, start_rack, orders, Vm_array_local, VU_map, D_racks, PROHIBITED_SLOT_INDICES):
    # Ensure module-level globals are available for the helper with the simpler signature
    prev_D_racks = globals().get('D_racks', None)
    prev_VU_map = globals().get('VU_map', None)
    prev_DISCHARGE_RACKS = globals().get('DISCHARGE_RACKS', None)
    prev_PROHIBITED = globals().get('PROHIBITED_SLOT_INDICES', None)
    globals()['D_racks'] = D_racks
    globals()['VU_map'] = VU_map
    globals()['DISCHARGE_RACKS'] = DISCHARGE_RACKS
    globals()['PROHIBITED_SLOT_INDICES'] = PROHIBITED_SLOT_INDICES
    try:
        augmented = insert_discharge_points_and_boxes(genome, slot_assignments, slot_to_rack_local, box_volume_max, start_rack, orders)
    finally:
        # restore previous globals to avoid side-effects
        if prev_D_racks is None:
            globals().pop('D_racks', None)
        else:
            globals()['D_racks'] = prev_D_racks
        if prev_VU_map is None:
            globals().pop('VU_map', None)
        else:
            globals()['VU_map'] = prev_VU_map
        if prev_DISCHARGE_RACKS is None:
            globals().pop('DISCHARGE_RACKS', None)
        else:
            globals()['DISCHARGE_RACKS'] = prev_DISCHARGE_RACKS
        if prev_PROHIBITED is None:
            globals().pop('PROHIBITED_SLOT_INDICES', None)
        else:
            globals()['PROHIBITED_SLOT_INDICES'] = prev_PROHIBITED
    total_distance = 0.0
    current = start_rack
    i = 2
    rutas_por_pedido = []
    ruta_actual = [start_rack]
    orders_arr = np.array(orders)
    order_demands_initial = [dict((int(sku)+1, int(qty)) for sku, qty in enumerate(order) if qty > 0) for order in orders_arr]
    order_demands = [d.copy() for d in order_demands_initial]
    # number of non-empty orders (ignore blank rows in input)
    expected_orders = sum(1 for d in order_demands_initial if any(q > 0 for q in d.values()))
    order_idx = 0

    while i < len(augmented):
        rack = augmented[i]
        if rack == 0:
            # if we already produced all expected orders, ignore trailing separators
            if order_idx >= expected_orders:
                break
            if current != start_rack:
                total_distance += D_racks[current, start_rack]
                current = start_rack
                ruta_actual.append(start_rack)
            rutas_por_pedido.append(ruta_actual[:])
            ruta_actual = [start_rack]
            order_idx += 1
            i += 1
            continue
        total_distance += D_racks[current, rack]
        current = rack
        ruta_actual.append(rack)
        if order_idx < len(order_demands):
            demand = order_demands[order_idx]
            for slot_idx, sku in enumerate(slot_assignments[genome[0]]):
                if slot_to_rack_local[slot_idx] != rack:
                    continue
                sku_id = int(sku)
                if sku_id == 0:
                    continue
                if sku_id in demand and demand[sku_id] > 0:
                    cap = capacity_of_slot(slot_idx, Vm_array_local, sku_id, VU_map)
                    take = min(demand[sku_id], cap)
                    demand[sku_id] -= take
        i += 1

    # append last route only if we haven't already reached expected_orders
    if ruta_actual != [start_rack] and order_idx < expected_orders:
        rutas_por_pedido.append(ruta_actual[:])

    penalized = False
    for d in order_demands:
        for qty in d.values():
            if qty > 0:
                penalized = True
                total_distance += 1e6
                break
        if penalized:
            break

    sku_dist = most_demanded_sku_distance(genome, slot_assignments, slot_to_rack_local, orders, start_rack, D_racks=D_racks)
    return float(total_distance), float(sku_dist), rutas_por_pedido, augmented

# ----------------------------
# --- Genetic operators for route-genomes
# ----------------------------
def order_crossover(parent1, parent2, px):
    if random.random() > px:
        chosen = random.choice([parent1, parent2])
        return {'genome': chosen['genome'][:], 'cluster_idx': chosen['cluster_idx']}
    g1, g2 = parent1['genome'], parent2['genome']
    if g1[0] != g2[0]:
        chosen = random.choice([parent1, parent2])
        return {'genome': chosen['genome'][:], 'cluster_idx': chosen['cluster_idx']}

    def split_orders(genome):
        orders, cur = [], []
        for gene in genome[1:]:
            if gene == 0:
                orders.append(cur)
                cur = []
            else:
                cur.append(gene)
        return orders

    orders1 = split_orders(g1)
    orders2 = split_orders(g2)
    if len(orders1) != len(orders2):
        chosen = random.choice([parent1, parent2])
        return {'genome': chosen['genome'][:], 'cluster_idx': chosen['cluster_idx']}

    child_orders = []
    for o1, o2 in zip(orders1, orders2):
        combined = list(dict.fromkeys(o1 + o2))
        random.shuffle(combined)
        child_orders.append(combined)
    new_genome = [g1[0], 0]
    for o in child_orders:
        new_genome += o + [0]
    return {'genome': new_genome, 'cluster_idx': g1[0]}

def swap_mutation(individual, pm):
    g = individual['genome'][:]
    zeros = [i for i, v in enumerate(g) if v == 0]
    for j in range(len(zeros)-1):
        start, end = zeros[j], zeros[j+1]
        if end - start > 2 and random.random() < pm:
            i1 = random.randint(start+1, end-2)
            i2 = random.randint(i1+1, end-1)
            g[i1], g[i2] = g[i2], g[i1]
    return {'genome': g, 'cluster_idx': individual['cluster_idx']}

# ----------------------------
# --- Non-dominated sort & crowding
# ----------------------------
def dominates_obj(p, q):
    return all(p_i <= q_i for p_i, q_i in zip(p, q)) and any(p_i < q_i for p_i, q_i in zip(p, q))

def fast_non_dominated_sort(objectives):
    pop_size = len(objectives)
    S = [[] for _ in range(pop_size)]
    n = [0] * pop_size
    fronts = [[]]
    for p in range(pop_size):
        for q in range(pop_size):
            if p == q: continue
            if dominates_obj(objectives[p], objectives[q]):
                S[p].append(q)
            elif dominates_obj(objectives[q], objectives[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts

def crowding_distance(front, objectives):
    distances = {i: 0.0 for i in front}
    if len(front) == 0:
        return distances
    n_obj = len(objectives[0])
    for m in range(n_obj):
        sorted_front = sorted(front, key=lambda i: objectives[i][m])
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')
        f_min = objectives[sorted_front[0]][m]
        f_max = objectives[sorted_front[-1]][m]
        if f_max == f_min:
            continue
        for k in range(1, len(sorted_front)-1):
            prev_val = objectives[sorted_front[k-1]][m]
            next_val = objectives[sorted_front[k+1]][m]
            distances[sorted_front[k]] += (next_val - prev_val) / (f_max - f_min)
    return distances

# ----------------------------
# --- NSGA-II loop for picking
# ----------------------------
def nsga2_picking_loop(orders, slot_assignment_list, Vm_array, VU_array,
                       DISCHARGE_RACKS_input, slot_to_rack_local, D_racks_array,
                       pop_size, n_gen, px, pm,
                       box_volume_max=1.0, start_rack=0, seed=None, verbose=True, PROHIBITED_SLOT_INDICES=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # localize globals used by helper functions
    global D_racks, VU_map, slot_to_rack, DISCHARGE_RACKS
    D_racks = D_racks_array
    slot_to_rack = slot_to_rack_local
    DISCHARGE_RACKS = list(DISCHARGE_RACKS_input)
    VU_map = {i+1: float(VU_array[i]) for i in range(len(VU_array))}

    if PROHIBITED_SLOT_INDICES is None:
        PROHIBITED_SLOT_INDICES = list(range(4))

    num_clusters = len(slot_assignment_list)
    # initialize population: seed with a genome per cluster
    population = []
    for cid, srow in enumerate(slot_assignment_list):
        g = build_genome(orders, srow, Vm_array, slot_to_rack, start_rack, cid, VU_map, D_racks, PROHIBITED_SLOT_INDICES)
        population.append({'genome': g, 'cluster_idx': cid})

    # fill remaining individuals randomly
    while len(population) < pop_size:
        cid = random.randint(0, num_clusters-1)
        srow = slot_assignment_list[cid]
        genome = [cid, 0]
        for order in orders:
            Vm_local = Vm_array.copy() if hasattr(Vm_array, "__len__") else np.array([Vm_array]*len(srow))
            indices = np.where(order > 0)[0]
            remaining = {int(idx+1): int(q) for idx, q in zip(indices, order[order>0])}
            racks_needed = set()
            for sku_id, qty in remaining.items():
                need = qty
                for slot_idx, slot_sku in enumerate(srow):
                    if int(slot_sku) != sku_id:
                        continue
                    if slot_idx in PROHIBITED_SLOT_INDICES:
                        continue
                    unit_vol = VU_map.get(sku_id, 0.0)
                    cap = int(Vm_local[slot_idx] // unit_vol) if unit_vol > 0 else 0
                    if cap <= 0:
                        continue
                    take = min(need, cap)
                    Vm_local[slot_idx] -= take * unit_vol
                    if take > 0:
                        racks_needed.add(slot_to_rack_local[slot_idx])
                        need -= take
                        if need <= 0:
                            break
            racks_needed = list(racks_needed)
            random.shuffle(racks_needed)
            genome += racks_needed + [0]
        population.append({'genome': genome, 'cluster_idx': cid})

    # store per-generation pareto fronts as tuples:
    # (list_of_deepcopied_inds, list_of_fitness_tuples, list_of_original_indices, generation_number)
    pareto_generations = []

    for gen in range(n_gen):
        # evaluate population
        population_eval = []
        for idx, ind in enumerate(population):
            d, sku_dist, rutas, aug = evaluate_individual(ind['genome'], slot_assignment_list, slot_to_rack,
                                                                    box_volume_max, start_rack, orders, Vm_array, VU_map, D_racks, PROHIBITED_SLOT_INDICES)
            population_eval.append((d, sku_dist, rutas, aug))
        population_obj = [(d, sku_dist) for (d, sku_dist, _, _) in population_eval]
        # attach objectives
        for idx, ind in enumerate(population):
            ind['objectives'] = population_obj[idx]

        # generate offspring
        offspring = []
        while len(offspring) < pop_size:
            a, b = random.sample(range(len(population)), 2)
            pa, pb = population[a], population[b]
            oa, ob = pa['objectives'], pb['objectives']
            if dominates_obj(oa, ob):
                parent1 = pa
            elif dominates_obj(ob, oa):
                parent1 = pb
            else:
                parent1 = random.choice([pa, pb])
            c, d_idx = random.sample(range(len(population)), 2)
            pc, pd = population[c], population[d_idx]
            oc, od = pc['objectives'], pd['objectives']
            if dominates_obj(oc, od):
                parent2 = pc
            elif dominates_obj(od, oc):
                parent2 = pd
            else:
                parent2 = random.choice([pc, pd])

            if parent1['cluster_idx'] == parent2['cluster_idx']:
                ch1 = order_crossover(parent1, parent2, px)
                ch2 = order_crossover(parent2, parent1, px)
            else:
                ch1 = swap_mutation(parent1, pm)
                ch2 = swap_mutation(parent2, pm)
            offspring.append(swap_mutation(ch1, pm))
            if len(offspring) < pop_size:
                offspring.append(swap_mutation(ch2, pm))

        # evaluate offspring & attach objectives
        offspring_eval = []
        for ind in offspring:
            d, sku_dist, rutas, aug = evaluate_individual(ind['genome'], slot_assignment_list, slot_to_rack,
                                                                    box_volume_max, start_rack, orders, Vm_array, VU_map, D_racks, PROHIBITED_SLOT_INDICES)
            offspring_eval.append((d, sku_dist, rutas, aug))
        for idx, ind in enumerate(offspring):
            ind['objectives'] = (offspring_eval[idx][0], offspring_eval[idx][1])

        # combine and select new population
        combined = population + offspring
        objectives = [ind['objectives'] for ind in combined]
        fronts_combined = fast_non_dominated_sort(objectives)
        new_pop = []
        for front in fronts_combined:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend([combined[i] for i in front])
            else:
                distances = crowding_distance(front, objectives)
                sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
                remaining = pop_size - len(new_pop)
                new_pop.extend([combined[i] for i in sorted_front[:remaining]])
                break
        population = new_pop

        # compute Pareto fronts for this population
        fronts = fast_non_dominated_sort([ind['objectives'] for ind in combined])
        front0 = fronts[0] if fronts else []
        pareto_inds = []
        pareto_fitness = []
        pareto_indices = []
        for i in front0:
            if i < len(population):
                pareto_inds.append(copy.deepcopy(population[i]))
                f = population[i]['objectives']
                pareto_fitness.append((float(f[0]), float(f[1])))
                pareto_indices.append(i)  # original index in this population
        if pareto_fitness:
            pareto_generations.append((pareto_inds, pareto_fitness, pareto_indices, gen))
        if verbose and (gen % max(1, n_gen//10) == 0):
            print(f"[gen {gen}] population={len(population)} pareto_size={len(pareto_fitness)}")

    # after all generations, pick best generation front by HV
    if not pareto_generations:
        if verbose:
            print("No Pareto fronts were collected during the run.")
        return population, None, None

    all_fitness_lists = [pf for (_, pf, _, _) in pareto_generations if len(pf) > 0]
    try:
        ref_point = make_ref_point_from_fronts(all_fitness_lists, factor=1.2, require_nondominated=True)
    except Exception:
        all_pts = np.vstack([np.array(f) for f in all_fitness_lists])
        ref_point = 1.2 * np.max(all_pts, axis=0)

    hv_values = []
    for inds, fitnesses, indices, gnum in pareto_generations:
        F = np.array(fitnesses, dtype=float)
        if F.size == 0:
            hv_values.append(0.0)
            continue
        F_nd = filter_nondominated(F)
        F_nd = F_nd[np.all(F_nd <= ref_point, axis=1)]
        hv_values.append(hv_2d_min(F_nd, ref_point))

    hv_values = np.array(hv_values, dtype=float)
    best_gen_idx = int(np.nanargmax(hv_values)) if hv_values.size > 0 else 0
    best_hv = float(hv_values[best_gen_idx]) if hv_values.size > 0 else 0.0
    best_inds, best_fitnesses, best_indices, best_gen_number = pareto_generations[best_gen_idx]

    if verbose:
      print(f"\nMejor frente encontrado: generación: {best_gen_number}")
      print(f"\n✅ Hipervolumen: {best_hv:.6f}")

    # Print all individuals in best front along with stored original index, fitness and augmented genome
    for j, (ind, fit, orig_idx) in enumerate(zip(best_inds, best_fitnesses, best_indices)):
        d, sku_dist, rutas, augmented = evaluate_individual(ind['genome'], slot_assignment_list, slot_to_rack,
                                                                       box_volume_max, start_rack, orders, Vm_array, VU_map, D_racks, PROHIBITED_SLOT_INDICES)
        print(f"\nBest front sol #{j} (original_index_in_population={orig_idx}, stored_generation={best_gen_number}):")
        print(f"  fitness: f1={fit[0]:.6f}, f2={fit[1]:.6f}")
        print(f"  cluster_idx: {ind['cluster_idx']}")
        print(f"  augmented genome (with discharge rack visits): {augmented}")
        print("  routes per order:")
        for pid, r in enumerate(rutas, start=1):
            print(f"    Order {pid}: {' -> '.join(map(str, r))}")

    # Plot best Pareto front with blue points only and no labels on the plot
    try:
        f1_vals = [f[0] for f in best_fitnesses]
        f2_vals = [f[1] for f in best_fitnesses]
        plt.figure(figsize=(7,5))
        plt.scatter(f1_vals, f2_vals, c='blue', s=60)  # blue points, no labels
        plt.xlabel("f1 (total distance)")
        plt.ylabel("f2 (distance to top SKUs)")
        plt.title(f"Best Pareto front (stored generation {best_gen_number})")
        plt.grid(True)
        plt.show()
    except Exception as e:
        print("Plot failed:", e)

    return population, best_inds, best_fitnesses

# ----------------------------
# --- High-level wrapper for Streamlit
# ----------------------------
def nsga2_picking_streamlit(slot_assignments, D, VU, Sr, D_racks,
                            pop_size=20, n_gen=10, px=1.0, pm=0.3, seed=None,
                            box_volume_max=1.0, start_rack=0, prohibited_slots=None, verbose=False):
    """
    slot_assignments: list/iterable of slot assignment arrays (each is a 1D array of SKU ids per slot)
    D: orders matrix (num_orders x num_skus)
    VU: dict or array with SKU volumes (1-based keys allowed)
    Sr: slot->rack matrix (num_slots x num_racks) or similar (we only use argmax per row)
    D_racks: distance matrix between racks

    Returns a dict or list of dicts with keys used by streamlit_app: 'pareto_front', 'population_eval_triples', 'distancia_total', 'sku_distancia', 'penalizado', 'rutas_best', 'augmented_best'
    """
    NUM_SKUS = int(D.shape[1])
    VU_array = _build_vu_array(VU, NUM_SKUS)
    slot_to_rack = np.argmax(Sr, axis=1).tolist()
    NUM_SLOTS = slot_assignments[0].shape[0]
    Vm = np.full(NUM_SLOTS, DEFAULT_VM_PER_SLOT)
    PROHIBITED_SLOT_INDICES = list(prohibited_slots) if prohibited_slots is not None else list(range(4))

    # Ensure DISCHARGE_RACKS derived from PROHIBITED slots and slot_to_rack
    discharge_racks = sorted(set(slot_to_rack[s] for s in PROHIBITED_SLOT_INDICES if 0 <= s < len(slot_to_rack)))
    # Normalize: exclude the start rack (don't treat start as a discharge rack)
    try:
        discharge_racks = [int(r) for r in discharge_racks if int(r) != start_rack]
    except Exception:
        discharge_racks = [r for r in discharge_racks if r != start_rack]
    # Fallback to a conservative default if empty — the user expects at least racks 1,2,3
    if not discharge_racks:
        discharge_racks = [1, 2, 3]

    results_combined = []
    # allow both a list of slotting solutions or a single 2D array
    if not isinstance(slot_assignments, (list, tuple)):
        slot_assignments = [slot_assignments]

    for idx, slot_assign in enumerate(slot_assignments):
        # run the NSGA-II picking loop for this set of slot assignments
        # DEBUG rápido: imprime información clave para diagnosticar por qué no se insertan discharge racks
        try:
            print("DEBUG: PROHIBITED_SLOT_INDICES =", PROHIBITED_SLOT_INDICES)
        except Exception as _:
            print("DEBUG: PROHIBITED_SLOT_INDICES not available")
        try:
            print("DEBUG: slot_to_rack (len) =", len(slot_to_rack), "sample first 10:", slot_to_rack[:10])
        except Exception as _:
            print("DEBUG: slot_to_rack not available or invalid")
        try:
            print("DEBUG: discharge_racks computed from prohibited slots =", discharge_racks)
        except Exception as _:
            print("DEBUG: discharge_racks not available")
        try:
            print("DEBUG: D_racks shape =", np.array(D_racks).shape)
        except Exception as e:
            print("DEBUG: D_racks not available or invalid:", e)

        population, best_inds, best_fitnesses = None, None, None
        population, best_inds, best_fitnesses = nsga2_picking_loop(
            orders=np.array(D),
            slot_assignment_list=[slot_assign],
            Vm_array=Vm,
            VU_array=VU_array,
            DISCHARGE_RACKS_input=discharge_racks,
            slot_to_rack_local=slot_to_rack,
            D_racks_array=np.array(D_racks),
            pop_size=pop_size,
            n_gen=n_gen,
            px=px,
            pm=pm,
            box_volume_max=box_volume_max,
            start_rack=start_rack,
            seed=seed,
            verbose=verbose,
            PROHIBITED_SLOT_INDICES=PROHIBITED_SLOT_INDICES
        )

        # nsga2_picking_loop returns population, best_inds, best_fitnesses (or best_inds may be None)
        if best_inds is None:
            # no pareto found, return minimal structure
            results_combined.append({'pareto_front': [], 'population_eval_triples': {}, 'best_hv': 0.0})
            continue

        # build mapping and extract augmented
        # best_inds is a list of individuals (each {'genome':..., 'cluster_idx':..., 'objectives':...})
        population_eval_triples = {}
        pareto_front = []
        rutas_best = []
        augmented_best = None
        # Normalize discharge racks returned to the caller: exclude start_rack
        # (we don't want to report the start rack as a discharge point) and
        # ensure a sensible fallback of [1,2,3] if the set becomes empty.
        try:
            discharge_racks_return = [int(r) for r in discharge_racks if int(r) != start_rack]
        except Exception:
            discharge_racks_return = [r for r in discharge_racks if r != start_rack] if discharge_racks else []
        if not discharge_racks_return:
            discharge_racks_return = [1, 2, 3]

        # Evaluate each best individual to get rutas and augmented (use evaluate_individual)
        for orig_idx, ind_obj, fit in zip(range(len(best_inds)), best_inds, best_fitnesses):
            # Determine genome to evaluate: prefer ind_obj['genome'] if valid, otherwise rebuild
            genome_to_eval = None
            try:
                # check that ind_obj is a dict-like with a genome that looks like a route-genome
                candidate = ind_obj.get('genome') if isinstance(ind_obj, dict) else None
                if candidate is not None and isinstance(candidate, (list, tuple)):
                    # Heuristic: a route-genome should have relatively few elements compared to NUM_SLOTS,
                    # and must start with cluster_idx and a 0 separator: [cluster_idx, 0, ...]
                    ok_prefix = (len(candidate) >= 2 and int(candidate[0]) == ind_obj.get('cluster_idx') and int(candidate[1]) == 0)
                    ok_length = len(candidate) < max(2, NUM_SLOTS // 2)  # route genomes are typically much shorter than NUM_SLOTS
                    if ok_prefix and ok_length:
                        genome_to_eval = list(candidate)
                # fallback: if not valid, rebuild genome using build_genome()
            except Exception:
                genome_to_eval = None

            if genome_to_eval is None:
                # rebuild using the slot_assign for this slotting solution (safe fallback)
                if verbose:
                    print(f"DEBUG: rebuilding genome for best individual idx={orig_idx} (ind_obj has no valid route-genome).")
                genome_to_eval = build_genome(np.array(D), slot_assign, Vm, slot_to_rack, start_rack, 0, {i+1: VU_array[i] for i in range(len(VU_array))}, np.array(D_racks), PROHIBITED_SLOT_INDICES)

            # Now evaluate
            d, sku_dist, rutas, augmented = evaluate_individual(genome_to_eval, [slot_assign], slot_to_rack,
                                                               box_volume_max, start_rack, np.array(D), Vm, {i+1: VU_array[i] for i in range(len(VU_array))}, np.array(D_racks), PROHIBITED_SLOT_INDICES)
            # store
            population_eval_triples[orig_idx] = (d, sku_dist, rutas, ind_obj, augmented)
            pareto_front.append((orig_idx, float(d), float(sku_dist)))
            rutas_best.append(rutas)
            if augmented_best is None:
                augmented_best = augmented

        # representative best (el primero del pareto_front)
        if pareto_front:
            first = pareto_front[0]
            rep_idx = first[0]
            rep_eval = population_eval_triples.get(rep_idx, (None, None, None, None, None))
            distancia_total = rep_eval[0]
            sku_distancia = rep_eval[1]
            rutas_best_flat = rep_eval[2]
            penalizado = any([True for v in rep_eval[2] if v is None]) if rep_eval[2] is not None else False
        else:
            distancia_total = None
            sku_distancia = None
            rutas_best_flat = []
            penalizado = False

        results_combined.append({
            'pareto_front': pareto_front,
            'population_eval_triples': population_eval_triples,
            'best_hv': None,
            'distancia_total': distancia_total,
            'sku_distancia': sku_distancia,
            'penalizado': penalizado,
            'rutas_best': rutas_best_flat,
            'augmented_best': augmented_best,
            'discharge_racks': discharge_racks_return,
            'slot_assignment_index': idx,
        })

    # if single input, return single dict for backward compat
    if len(results_combined) == 1:
        return results_combined[0]
    return results_combined

if __name__ == "__main__":
    print("picking_solver module loaded. Use nsga2_picking_streamlit(...) from your code.")