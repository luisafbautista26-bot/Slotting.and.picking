"""Picking solver - full implementation following user's pseudocode.

This module implements HV utilities for 2D minimization, volume/capacity
calculations, route builders (nearest-first), insertion of discharge points
and a compact NSGA-II loop. Defaults applied:
- PROHIBITED_SLOT_INDICES = [0,1,2,3]
- DEFAULT_DISCHARGE_RACKS = [0,1,2,3,4]
- NaN distances in D_racks are replaced with a large value (1e6)

The implementation is intentionally defensive (bounds checks and NaN
handling) so it can run with the sample data in the repo.
"""

from typing import List, Dict, Iterable, Tuple
import copy
import random
import numpy as np

DEFAULT_VM_PER_SLOT = 3
PROHIBITED_SLOT_INDICES = list(range(4))
DEFAULT_DISCHARGE_RACKS = [0, 1, 2, 3, 4]


def is_dominated(p, q):
    return (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1])


def filter_nondominated(points: Iterable[Tuple[float, float]]) -> np.ndarray:
    pts = np.array(list(points), dtype=float)
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


def hv_2d_min(points: Iterable[Tuple[float, float]], ref: Iterable[float]) -> float:
    pts = np.array(list(points), dtype=float)
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
    return float(hv)


def make_ref_point_from_fronts(fronts: List[Iterable[Tuple[float, float]]], factor=1.2, require_nondominated=True) -> np.ndarray:
    valid_fronts = [np.array(f, dtype=float) for f in fronts if len(f) > 0]
    if not valid_fronts:
        raise ValueError("No fronts to build ref_point.")
    all_pts = np.vstack(valid_fronts)
    if require_nondominated:
        all_pts = filter_nondominated(all_pts)
    if all_pts.size == 0:
        raise ValueError("No valid points for ref_point.")
    return factor * np.max(all_pts, axis=0)


def _build_vu_map(VU_array: Iterable) -> Dict[int, float]:
    if VU_array is None:
        return {}
    if isinstance(VU_array, dict):
        return {int(k): float(v) for k, v in VU_array.items()}
    arr = list(VU_array)
    return {i + 1: float(arr[i]) for i in range(len(arr))}


def capacity_of_slot(slot_idx: int, Vm_array, sku_id: int, VU_map: Dict[int, float]) -> int:
    if sku_id == 0:
        return 0
    Vm_slot = Vm_array[slot_idx] if hasattr(Vm_array, "__len__") else Vm_array
    unit_vol = VU_map.get(sku_id, 0.0)
    if unit_vol <= 0:
        return 0
    return int(Vm_slot // unit_vol)


def route_for_order(order, slot_assignment_row, Vm_array_local, slot_to_rack_local, start_rack, VU_map, D_racks):
    # Greedy selection of racks that contain needed SKUs (respecting prohibited slots)
    Vm_local = Vm_array_local.copy() if hasattr(Vm_array_local, "__len__") else np.array([Vm_array_local] * len(slot_assignment_row))
    route = [start_rack]
    indices = np.where(np.array(order) > 0)[0]
    demandas = order[np.array(order) > 0].astype(int)
    remaining = {int(idx + 1): int(q) for idx, q in zip(indices, demandas)}
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

    # visit racks by nearest-first from start
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


def build_genome(orders, slot_assignment_row, Vm_array_local, slot_to_rack_local, start_rack, cluster_idx, VU_map, D_racks):
    genome = [cluster_idx, 0]
    for order in orders:
        sub = route_for_order(order, slot_assignment_row, Vm_array_local, slot_to_rack_local, start_rack, VU_map, D_racks)
        genome += sub[1:]
    return genome


def insert_discharge_points_and_boxes(genome, slot_assignments, slot_to_rack_local,
                                      box_volume_max, start_rack, orders, VU_map, D_racks):
    cluster_idx = genome[0]
    slot_assignment_row = slot_assignments[cluster_idx]
    new_genome = [cluster_idx, 0]
    i = 2
    order_idx = 0
    current_rack = start_rack
    box_vol = 0.0
    orders_arr = np.array(orders)
    order_demands = [dict((int(sku) + 1, int(qty)) for sku, qty in enumerate(order) if qty > 0) for order in orders_arr]
    num_orders = len(order_demands)

    while i < len(genome):
        rack = genome[i]
        if rack == 0:
            # separator: ensure a discharge point before separator if needed
            if new_genome[-1] not in DEFAULT_DISCHARGE_RACKS:
                nearest_discharge = min(DEFAULT_DISCHARGE_RACKS, key=lambda dp: D_racks[current_rack, dp])
                new_genome.append(nearest_discharge)
                current_rack = nearest_discharge
                box_vol = 0.0
            new_genome.append(0)
            order_idx += 1
            current_rack = start_rack
            box_vol = 0.0
            i += 1
            continue

        new_genome.append(rack)
        current_rack = rack

        if order_idx < num_orders:
            demand = order_demands[order_idx]
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
                        nearest_discharge = min(DEFAULT_DISCHARGE_RACKS, key=lambda dp: D_racks[current_rack, dp])
                        new_genome.append(nearest_discharge)
                        current_rack = nearest_discharge
                        box_vol = 0.0
                    box_vol += unit_vol
                demand[sku_id] = 0
        i += 1

    # remove consecutive duplicate discharge points
    filtered = [new_genome[0], new_genome[1]]
    for j in range(2, len(new_genome)):
        if new_genome[j] in DEFAULT_DISCHARGE_RACKS and filtered[-1] == new_genome[j]:
            continue
        filtered.append(new_genome[j])
    return filtered


def most_demanded_sku_distance(genome, slot_assignments, slot_to_rack_local, orders, start_rack=0, top_k=5):
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
    return float(total)


def evaluate_individual(genome, slot_assignments, slot_to_rack_local, box_volume_max, start_rack, orders, Vm_array_local, VU_map, D_racks):
    augmented = insert_discharge_points_and_boxes(genome, slot_assignments, slot_to_rack_local, box_volume_max, start_rack, orders, VU_map, D_racks)
    total_distance = 0.0
    current = start_rack
    i = 2
    rutas_por_pedido = []
    ruta_actual = [start_rack]
    orders_arr = np.array(orders)
    order_demands_initial = [dict((int(sku) + 1, int(qty)) for sku, qty in enumerate(order) if qty > 0) for order in orders_arr]
    order_demands = [d.copy() for d in order_demands_initial]
    order_idx = 0

    while i < len(augmented):
        rack = augmented[i]
        if rack == 0:
            if current != start_rack:
                total_distance += float(D_racks[current, start_rack])
                current = start_rack
                ruta_actual.append(start_rack)
            rutas_por_pedido.append(ruta_actual[:])
            ruta_actual = [start_rack]
            order_idx += 1
            i += 1
            continue
        total_distance += float(D_racks[current, rack])
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

    if ruta_actual != [start_rack]:
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

    sku_dist = most_demanded_sku_distance(genome, slot_assignments, slot_to_rack_local, orders, start_rack)
    return float(total_distance), float(sku_dist), bool(penalized), rutas_por_pedido, augmented


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
    for j in range(len(zeros) - 1):
        start, end = zeros[j], zeros[j + 1]
        if end - start > 2 and random.random() < pm:
            i1 = random.randint(start + 1, end - 2)
            i2 = random.randint(i1 + 1, end - 1)
            g[i1], g[i2] = g[i2], g[i1]
    return {'genome': g, 'cluster_idx': individual['cluster_idx']}


def dominates_obj(p, q):
    return all(p_i <= q_i for p_i, q_i in zip(p, q)) and any(p_i < q_i for p_i, q_i in zip(p, q))


def fast_non_dominated_sort(objectives):
    pop_size = len(objectives)
    S = [[] for _ in range(pop_size)]
    n = [0] * pop_size
    fronts = [[]]
    for p in range(pop_size):
        for q in range(pop_size):
            if p == q:
                continue
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
        for k in range(1, len(sorted_front) - 1):
            prev_val = objectives[sorted_front[k - 1]][m]
            next_val = objectives[sorted_front[k + 1]][m]
            distances[sorted_front[k]] += (next_val - prev_val) / (f_max - f_min)
    return distances


def nsga2_picking_loop(orders, slot_assignment_list, Vm_array, VU_array,
                       DISCHARGE_RACKS_input, slot_to_rack_local, D_racks_array,
                       pop_size=50, n_gen=100, px=0.8, pm=0.2,
                       box_volume_max=1.0, start_rack=0, seed=None, verbose=True):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    D_racks_clean = np.nan_to_num(np.array(D_racks_array), nan=1e6)
    VU_map = _build_vu_map(VU_array)

    num_clusters = len(slot_assignment_list)
    population = []
    for cid, srow in enumerate(slot_assignment_list):
        g = build_genome(orders, srow, Vm_array, slot_to_rack_local, start_rack, cid, VU_map, D_racks_clean)
        population.append({'genome': g, 'cluster_idx': cid})

    while len(population) < pop_size:
        cid = random.randint(0, max(0, num_clusters - 1))
        srow = slot_assignment_list[cid]
        genome = [cid, 0]
        for order in orders:
            Vm_local = Vm_array.copy() if hasattr(Vm_array, '__len__') else np.array([Vm_array] * len(srow))
            indices = np.where(np.array(order) > 0)[0]
            remaining = {int(idx + 1): int(q) for idx, q in zip(indices, order[np.array(order) > 0])}
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

    pareto_generations = []
    for gen in range(n_gen):
        population_eval = []
        for idx, ind in enumerate(population):
            d, sku_dist, penalized, rutas, aug = evaluate_individual(ind['genome'], slot_assignment_list, slot_to_rack_local, box_volume_max, start_rack, orders, Vm_array, VU_map, D_racks_clean)
            population_eval.append((d, sku_dist, penalized, rutas, aug))
        population_obj = [(d, sku_dist, penalized) for (d, sku_dist, penalized, _, _) in population_eval]
        for idx, ind in enumerate(population):
            ind['objectives'] = population_obj[idx]

        fronts = fast_non_dominated_sort([ind['objectives'] for ind in population])
        front0 = fronts[0] if fronts else []
        pareto_inds = []
        pareto_fitness = []
        pareto_indices = []
        for i in front0:
            if i < len(population):
                pareto_inds.append(copy.deepcopy(population[i]))
                f = population[i]['objectives']
                pareto_fitness.append((float(f[0]), float(f[1])))
                pareto_indices.append(i)
        if pareto_fitness:
            pareto_generations.append((pareto_inds, pareto_fitness, pareto_indices, gen))
        if verbose and (gen % max(1, n_gen // 10) == 0):
            print(f"[gen {gen}] population={len(population)} pareto_size={len(pareto_fitness)}")

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

        offspring_eval = []
        for ind in offspring:
            d, sku_dist, penalized, rutas, aug = evaluate_individual(ind['genome'], slot_assignment_list, slot_to_rack_local, box_volume_max, start_rack, orders, Vm_array, VU_map, D_racks_clean)
            offspring_eval.append((d, sku_dist, penalized, rutas, aug))
        for idx, ind in enumerate(offspring):
            ind['objectives'] = (offspring_eval[idx][0], offspring_eval[idx][1], offspring_eval[idx][2])

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
        print(f"\nBest generation: {best_gen_number}")
        print(f"Hypervolume: {best_hv:.6f}")

    return population, best_inds, best_fitnesses


def nsga2_picking_streamlit(slot_assignments, D, VU_array, Sr, D_racks_array, pop_size=20, n_gen=10, prohibited_slots=None):
    slot_assignments_list = [np.asarray(sa) for sa in slot_assignments]
    NUM_SLOTS = int(slot_assignments_list[0].shape[0]) if len(slot_assignments_list) > 0 else 0
    Vm = np.full(NUM_SLOTS, DEFAULT_VM_PER_SLOT)
    slot_to_rack_local = np.argmax(Sr, axis=1).tolist() if Sr is not None else [0] * NUM_SLOTS
    prohibited = list(prohibited_slots) if prohibited_slots is not None else PROHIBITED_SLOT_INDICES
    discharge_racks = DEFAULT_DISCHARGE_RACKS
    VU_map = _build_vu_map(VU_array)
    D_racks_clean = np.nan_to_num(np.array(D_racks_array), nan=1e6)

    results = []
    for slot_assignment in slot_assignments_list:
        pop, best_inds, best_fitnesses = nsga2_picking_loop(
            orders=D,
            slot_assignment_list=[slot_assignment],
            Vm_array=Vm,
            VU_array=VU_array,
            DISCHARGE_RACKS_input=discharge_racks,
            slot_to_rack_local=slot_to_rack_local,
            D_racks_array=D_racks_clean,
            pop_size=pop_size,
            n_gen=n_gen,
            px=0.8,
            pm=0.2,
            box_volume_max=1.0,
            start_rack=0,
            seed=None,
            verbose=False,
        )

        population_eval_full = []
        if best_inds is not None:
            for indiv in best_inds:
                genome = indiv['genome']
                d, sku_dist, penalized, rutas, augmented = evaluate_individual(genome, [slot_assignment], slot_to_rack_local, 1.0, 0, D, Vm, VU_map, D_racks_clean)
                population_eval_full.append((d, sku_dist, penalized, rutas, augmented))

        pf = [(i, f[0], f[1]) for i, f in enumerate(best_fitnesses)] if best_fitnesses is not None else []

        results.append({
            'pareto_front': pf,
            'population_eval_triples': population_eval_full,
            'distancia_total': population_eval_full[0][0] if population_eval_full else None,
            'sku_distancia': population_eval_full[0][1] if population_eval_full else None,
            'penalizado': population_eval_full[0][2] if population_eval_full else True,
            'rutas_best': population_eval_full[0][3] if population_eval_full else [],
            'augmented_best': population_eval_full[0][4] if population_eval_full else [],
        })

    return results
"""Minimal clean picking_solver stub for import testing.

This file is intentionally small and safe. It will be replaced later by the
full implementation once the workspace is stable.
"""

import numpy as np
from typing import List, Dict

PROHIBITED_SLOT_INDICES = list(range(4))
DEFAULT_DISCHARGE_RACKS = [0, 1, 2, 3, 4]


def _build_vu_map(VU_array):
    if VU_array is None:
        return {}
    if isinstance(VU_array, dict):
        return {int(k): float(v) for k, v in VU_array.items()}
    arr = list(VU_array)
    return {i + 1: float(arr[i]) for i in range(len(arr))}


def nsga2_picking_streamlit(slot_assignments, D, VU_array, Sr, D_racks_array, pop_size=20, n_gen=10, prohibited_slots=None):
    """Very small stub: returns empty results but is import-safe."""
    # sanitize D_racks
    D_racks_clean = np.nan_to_num(np.array(D_racks_array), nan=1e6)
    VU_map = _build_vu_map(VU_array)
    results = []
    for sa in slot_assignments:
        results.append({'pareto_front': [], 'population_eval_triples': [], 'distancia_total': None})
    return results
