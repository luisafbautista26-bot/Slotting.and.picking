import numpy as np
import random
import copy
import matplotlib.pyplot as plt

__all__ = ["nsga2_picking_streamlit"]

# Default Vm (volume capacity per slot)
DEFAULT_VM_PER_SLOT = 3

# ----------------------------
# HV utilities (2D minimization)
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
# Picking / genome logic
# ----------------------------

# Globals set in the loop
D_racks = None
VU_map = {}
DISCHARGE_RACKS = []
slot_to_rack = []
PROHIBITED_SLOT_INDICES = []


def capacity_of_slot(slot_idx, Vm_array, sku_id):
    if sku_id == 0:
        return 0
    Vm_slot = Vm_array[slot_idx] if hasattr(Vm_array, "__len__") else Vm_array
    unit_vol = VU_map.get(sku_id, 0.0)
    if unit_vol <= 0:
        return 0
    return int(Vm_slot // unit_vol)


def route_for_order(order, slot_assignment_row, Vm_array_local, slot_to_rack_local, start_rack):
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


def build_genome(orders, slot_assignment_row, Vm_array_local, slot_to_rack_local, start_rack, cluster_idx):
    genome = [cluster_idx, 0]
    for order in orders:
        sub = route_for_order(order, slot_assignment_row, Vm_array_local, slot_to_rack_local, start_rack)
        genome += sub[1:]
    return genome


def insert_discharge_points_and_boxes(genome, slot_assignments, slot_to_rack_local,
                                      box_volume_max, start_rack, orders):
    cluster_idx = genome[0]
    slot_assignment_row = slot_assignments[cluster_idx]
    new_genome = [cluster_idx, 0]
    i = 2
    order_idx = 0
    current_rack = start_rack
    box_vol = 0.0
    orders_arr = np.array(orders)
    order_demands = [dict((int(sku)+1, int(qty)) for sku, qty in enumerate(order) if qty > 0) for order in orders_arr]
    num_orders = len(order_demands)
    # flag: whether we've picked any item since the last discharge
    pickups_since_last = False

    while i < len(genome):
        rack = genome[i]
        if rack == 0:
            # before closing order, insert nearest discharge if we picked since last discharge
            if pickups_since_last and DISCHARGE_RACKS:
                # avoid inserting duplicate discharge if already last
                if new_genome[-1] not in DISCHARGE_RACKS:
                    nearest_discharge = min(DISCHARGE_RACKS, key=lambda dp: D_racks[current_rack, dp])
                    new_genome.append(nearest_discharge)
                    current_rack = nearest_discharge
                    box_vol = 0.0
                # after discharging, reset pickup flag
                pickups_since_last = False
            # append order separator
            new_genome.append(0)
            order_idx += 1
            current_rack = start_rack
            box_vol = 0.0
            i += 1
            continue

        # visit this rack
        new_genome.append(rack)
        current_rack = rack

        # simulate picks at this rack: scan slots that belong to this rack
        picks_this_rack = 0
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
                # pick items one by one, discharging when needed
                for _ in range(qty_to_pick):
                    if unit_vol <= 0:
                        # cannot pick this SKU (volume zero), skip
                        continue
                    if box_vol + unit_vol > box_volume_max:
                        # need to discharge before picking
                        if DISCHARGE_RACKS:
                            # avoid duplicate consecutive discharge
                            if new_genome[-1] not in DISCHARGE_RACKS:
                                nearest_discharge = min(DISCHARGE_RACKS, key=lambda dp: D_racks[current_rack, dp])
                                new_genome.append(nearest_discharge)
                                current_rack = nearest_discharge
                                box_vol = 0.0
                                # after discharge, we haven't picked yet
                                pickups_since_last = False
                        else:
                            # no discharge defined: reset box anyway
                            box_vol = 0.0
                    # pick into box
                    box_vol += unit_vol
                    picks_this_rack += 1
                # mark demand as satisfied at this rack
                demand[sku_id] = 0
        # if we picked anything at this rack, mark that we've had pickups since last discharge
        if picks_this_rack > 0:
            pickups_since_last = True
        i += 1

    filtered = [new_genome[0], new_genome[1]]
    for j in range(2, len(new_genome)):
        if new_genome[j] in DISCHARGE_RACKS and filtered[-1] == new_genome[j]:
            continue
        filtered.append(new_genome[j])
    return filtered


def most_demanded_sku_distance(genome, slot_assignments, slot_to_rack_local, orders, start_rack=0, top_k=5):
    cluster_idx = genome[0]
    slot_assignment_row = slot_assignments[cluster_idx]
    sku_demands = np.sum(orders, axis=0)
    nonzero = np.where(sku_demands > 0)[0]
    top = nonzero[np.argsort(sku_demands[nonzero])[::-1][:top_k]]
    total = 0.0
    for sku_idx in top:
        sku_id = int(sku_idx) + 1
        slots = [i for i, s in enumerate(slot_assignment_row) if int(s) == sku_id]
        for slot in slots:
            rack = slot_to_rack_local[slot]
            total += D_racks[start_rack, rack]
    return total


def evaluate_individual(genome, slot_assignments, slot_to_rack_local, box_volume_max, start_rack, orders, Vm_array_local):
    augmented = insert_discharge_points_and_boxes(genome, slot_assignments, slot_to_rack_local, box_volume_max, start_rack, orders)
    total_distance = 0.0
    current = start_rack
    i = 2
    rutas_por_pedido = []
    ruta_actual = [start_rack]
    orders_arr = np.array(orders)
    order_demands_initial = [dict((int(sku)+1, int(qty)) for sku, qty in enumerate(order) if qty > 0) for order in orders_arr]
    order_demands = [d.copy() for d in order_demands_initial]
    order_idx = 0

    while i < len(augmented):
        rack = augmented[i]
        if rack == 0:
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
                    cap = capacity_of_slot(slot_idx, Vm_array_local, sku_id)
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
    return float(total_distance), float(sku_dist), penalized, rutas_por_pedido, augmented


# ----------------------------
# Genetic operators for route-genomes
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
# Non-dominated sort & crowding
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
# NSGA-II loop - stores full individuals and original indices
# ----------------------------
def nsga2_picking_loop(orders, slot_assignment_list, Vm_array, VU_array,
                       DISCHARGE_RACKS_input, slot_to_rack_local, D_racks_array,
                       pop_size, n_gen, px, pm,
                       box_volume_max=1.0, start_rack=0, seed=None, verbose=True):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    global D_racks, VU_map, slot_to_rack, DISCHARGE_RACKS, PROHIBITED_SLOT_INDICES
    D_racks = D_racks_array
    slot_to_rack = slot_to_rack_local
    DISCHARGE_RACKS = DISCHARGE_RACKS_input
    VU_map = {i+1: float(VU_array[i]) for i in range(len(VU_array))}

    # Prohibited slots: first 4 slots
    PROHIBITED_SLOT_INDICES = list(range(4))

    num_clusters = len(slot_assignment_list)
    population = []
    for cid, srow in enumerate(slot_assignment_list):
        g = build_genome(orders, srow, Vm_array, slot_to_rack, start_rack, cid)
        population.append({'genome': g, 'cluster_idx': cid})

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

    pareto_generations = []

    for gen in range(n_gen):
        population_eval = []
        for idx, ind in enumerate(population):
            d, sku_dist, penalized, rutas, aug = evaluate_individual(ind['genome'], slot_assignment_list, slot_to_rack,
                                                                    box_volume_max, start_rack, orders, Vm_array)
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
        if verbose and (gen % max(1, n_gen//10) == 0):
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
            d, sku_dist, penalized, rutas, aug = evaluate_individual(ind['genome'], slot_assignment_list, slot_to_rack,
                                                                    box_volume_max, start_rack, orders, Vm_array)
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
      print(f"\nMejor frente encontrado: generación: {best_gen_number}")
      print(f"\n✅ Hipervolumen: {best_hv:.6f}")

    for j, (ind, fit, orig_idx) in enumerate(zip(best_inds, best_fitnesses, best_indices)):
        d, sku_dist, penalized, rutas, augmented = evaluate_individual(ind['genome'], slot_assignment_list, slot_to_rack,
                                                                       box_volume_max, start_rack, orders, Vm_array)
        print(f"\nBest front sol #{j} (original_index_in_population={orig_idx}, stored_generation={best_gen_number}):")
        print(f"  fitness: f1={fit[0]:.6f}, f2={fit[1]:.6f}, penalized={penalized}")
        print(f"  cluster_idx: {ind['cluster_idx']}")
        print(f"  augmented genome (with discharge rack visits): {augmented}")
        print("  routes per order:")
        for pid, r in enumerate(rutas, start=1):
            print(f"    Order {pid}: {' -> '.join(map(str, r))}")

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
# Wrapper for Streamlit
# ----------------------------

def initialize_population(orders, slot_assignment_list, Vm, VU, slot_to_rack_local, start_rack, pop_size, D_racks_array):
    """Crear una población inicial simple a partir de las soluciones de slotting.
    Esta función establece algunas variables globales necesarias por las funciones
    de evaluación y genera genomas base para cada cluster (slot_assignment).
    """
    global D_racks, VU_map, slot_to_rack, PROHIBITED_SLOT_INDICES
    D_racks = D_racks_array
    slot_to_rack = slot_to_rack_local
    PROHIBITED_SLOT_INDICES = list(range(4))
    # VU puede venir como dict o array
    if hasattr(VU, '__len__'):
        VU_map = {i+1: float(VU[i]) for i in range(len(VU))}
    elif isinstance(VU, dict):
        VU_map = {int(k): float(v) for k, v in VU.items()}
    else:
        VU_map = {}

    population = []
    num_clusters = len(slot_assignment_list)
    for cid, srow in enumerate(slot_assignment_list):
        g = build_genome(orders, srow, Vm, slot_to_rack, start_rack, cid)
        population.append({'genome': g, 'cluster_idx': cid})

    # Rellenar población con genomas aleatorios basados en racks necesarios por pedido
    while len(population) < pop_size:
        cid = random.randint(0, max(0, num_clusters-1))
        srow = slot_assignment_list[cid]
        genome = [cid, 0]
        for order in orders:
            Vm_local = Vm.copy() if hasattr(Vm, '__len__') else np.array([Vm]*len(srow))
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
                    if unit_vol <= 0:
                        continue
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

    return population


def nsga2_picking_streamlit(slot_assignments, D, VU_array, Sr, D_racks_array, pop_size=20, n_gen=10):
    """
    slot_assignments: list of arrays (each is a slot_assignment solution)
    D: orders matrix (num_orders x num_skus)
    VU_array: 1D array of unit volumes per SKU (index 0 -> sku 1)
    Sr: slots x racks matrix (one-hot)
    D_racks_array: racks x racks distance matrix
    """
    results = []
    for idx, slot_assignment in enumerate(slot_assignments):
        NUM_SLOTS = slot_assignment.shape[0]
        Vm = np.full(NUM_SLOTS, DEFAULT_VM_PER_SLOT)
        slot_to_rack_local = np.argmax(Sr, axis=1).tolist()
        prohibited = list(range(4))
        discharge_racks = sorted(set(slot_to_rack_local[s] for s in prohibited))

        # Inicializar variables globales para las funciones auxiliares
        global D_racks, VU_map, slot_to_rack, DISCHARGE_RACKS, PROHIBITED_SLOT_INDICES
        D_racks = D_racks
        slot_to_rack = slot_to_rack_local
        PROHIBITED_SLOT_INDICES = prohibited
        if hasattr(VU, '__len__'):
            VU_map = {i+1: float(VU[i]) for i in range(len(VU))}
        elif isinstance(VU, dict):
            VU_map = {int(k): float(v) for k, v in VU.items()}
        else:
            VU_map = {}
        DISCHARGE_RACKS = discharge_racks

        pop, best_inds, best_fitnesses = nsga2_picking_loop(
            orders=D,
            slot_assignment_list=[slot_assignment],
            Vm_array=Vm,
            VU_array=VU_array,
            DISCHARGE_RACKS_input=discharge_racks,
            slot_to_rack_local=slot_to_rack_local,
            D_racks_array=D_racks_array,
            pop_size=pop_size,
            n_gen=n_gen,
            px=0.8,
            pm=0.95,
            box_volume_max=1.0,
            start_rack=0,
            seed=None,
            verbose=False,
        )

        if best_inds is None:
            results.append({'pareto_front': [], 'fig': None, 'population_eval_triples': [], 'rutas_best': [], 'distancia_total': None, 'sku_distancia': None, 'penalizado': True})
            continue

        population_eval_full = []
        for ind in best_inds:
            d, sku_dist, penalized, rutas, augmented = evaluate_individual(ind['genome'], [slot_assignment], slot_to_rack_local, 1.0, 0, D, Vm)
            population_eval_full.append((d, sku_dist, penalized, rutas))

        pf = [(i, f[0], f[1]) for i, f in enumerate(best_fitnesses)]

        fig = None
        try:
            fig, ax = plt.subplots(figsize=(7,5))
            xs = [f[0] for f in best_fitnesses]
            ys = [f[1] for f in best_fitnesses]
            ax.scatter(xs, ys, color='tab:blue', alpha=0.7)
            if best_fitnesses:
                best_idx = int(np.argmin([f[0] for f in best_fitnesses]))
                ax.scatter([best_fitnesses[best_idx][0]], [best_fitnesses[best_idx][1]], color='gold', marker='*', s=200, edgecolor='black')
            ax.set_xlabel('Distancia Total')
            ax.set_ylabel('Dist a SKUs demandados')
            ax.set_title(f'Pareto Picking - Slotting {idx+1}')
            ax.grid(True)
        except Exception:
            fig = None

        best_idx = 0
        if population_eval_full:
            best_idx = min(range(len(population_eval_full)), key=lambda i: population_eval_full[i][0])
            rutas_best = population_eval_full[best_idx][3]
        else:
            rutas_best = []

        results.append({
            'pareto_front': pf,
            'fig': fig,
            'population_eval_triples': population_eval_full,
            'rutas_best': rutas_best,
            'distancia_total': population_eval_full[best_idx][0] if population_eval_full else None,
            'sku_distancia': population_eval_full[best_idx][1] if population_eval_full else None,
            'penalizado': population_eval_full[best_idx][2] if population_eval_full else True,
        })

    return results
    while i < len(genome_with_discharges):
        rack = genome_with_discharges[i]
        if rack == 0:
            if current_rack != start_rack:
                total_distance += D_racks[current_rack, start_rack]
                current_rack = start_rack
                ruta_actual.append(start_rack)
            rutas_por_pedido.append(ruta_actual[:])
            ruta_actual = [start_rack]
            order_idx += 1
            i += 1
            continue
        total_distance += D_racks[current_rack, rack]
        current_rack = rack
        ruta_actual.append(rack)
        i += 1
    for demand in order_demands:
        for qty_left in demand.values():
            if qty_left > 0:
                penalized = True
                total_distance += 1e6
                break
    return total_distance, 0, penalized, rutas_por_pedido

def order_crossover(parent1, parent2):
    g1, g2 = parent1['genome'], parent2['genome']
    cluster_idx = g1[0]
    assert cluster_idx == g2[0], "Crossover only between same cluster"
    def split_orders(genome):
        orders, current = [], []
        for gene in genome[1:]:
            if gene == 0:
                if current:
                    orders.append(current)
                    current = []
            else:
                current.append(gene)
        return orders
    orders1 = split_orders(g1)
    orders2 = split_orders(g2)
    if len(orders1) != len(orders2):
        return swap_mutation(parent1)
    n_orders = len(orders1)
    child_orders = []
    for o1, o2 in zip(orders1, orders2):
        combined_racks = list(set(o1) | set(o2))
        random.shuffle(combined_racks)
        child_orders.append(combined_racks)
    new_genome = [cluster_idx, 0]
    for o in child_orders:
        new_genome += o + [0]
    child = {'genome': new_genome, 'cluster_idx': cluster_idx}
    return child

def swap_mutation(individual, mutation_rate=0.5):
    g = individual['genome'][:]
    indices = [i for i, v in enumerate(g) if v == 0]
    for j in range(len(indices)-1):
        start, end = indices[j], indices[j+1]
        if end - start > 2 and random.random() < mutation_rate:
            i1 = random.randint(start+1, end-2)
            i2 = random.randint(i1+1, end-1)
            g[i1], g[i2] = g[i2], g[i1]
    mutated = {'genome': g, 'cluster_idx': individual['cluster_idx']}
    return mutated

def crowding_distance(front, objectives):
    distances = {i: 0.0 for i in front}
    if len(front) == 0:
        return distances
    n_obj = len(objectives[0])
    for m in range(n_obj):
        sorted_front = sorted(front, key=lambda i: objectives[i][m])
        f_min = objectives[sorted_front[0]][m]
        f_max = objectives[sorted_front[-1]][m]
        distances[sorted_front[0]] = float('inf')
        distances[sorted_front[-1]] = float('inf')
        if f_max == f_min:
            continue
        for k in range(1, len(sorted_front) - 1):
            prev_val = objectives[sorted_front[k - 1]][m]
            next_val = objectives[sorted_front[k + 1]][m]
            distances[sorted_front[k]] += (next_val - prev_val) / (f_max - f_min)
    return distances

def pareto_front(population_eval_triples):
    pf = []
    for i, (dist, sku_dist, penalized) in enumerate(population_eval_triples):
        dominated = False
        for j, (d2, b2, p2) in enumerate(population_eval_triples):
            if (d2 < dist and b2 <= sku_dist) or (d2 <= dist and b2 < sku_dist):
                dominated = True
                break
        if not dominated:
            pf.append((i, dist, sku_dist))
    return pf

def fast_non_dominated_sort(objectives):
    pop_size = len(objectives)
    S = [[] for _ in range(pop_size)]
    n = [0] * pop_size
    rank = [0] * pop_size
    fronts = [[]]
    def dominates(p, q):
        return all(p_i <= q_i for p_i, q_i in zip(p, q)) and any(p_i < q_i for p_i, q_i in zip(p, q))
    for p in range(pop_size):
        for q in range(pop_size):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                S[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts

def nsga2_picking_streamlit(slot_assignments, D, VU, Sr, D_racks, pop_size=20, n_gen=10):
    """
    slot_assignments: lista de arrays (cada uno es una solución de slotting)
    D: matriz de pedidos (num_pedidos x num_skus)
    VU: dict de volumen unitario por SKU
    Sr: matriz de slots x racks (1 si el slot pertenece al rack)
    D_racks: matriz de distancias entre racks
    """
    resultados = []
    NUM_SLOTS = slot_assignments[0].shape[0]
    Vm = np.full(NUM_SLOTS, DEFAULT_VM_PER_SLOT)
    START_RACK = 0
    DISCHARGE_POINTS = [1, 2, 3]
    orders = np.array(D)
    slot_to_rack = np.argmax(Sr, axis=1).tolist()
    Vc = 1

    # Inicializar globals usados por funciones auxiliares
    global VU_map, DISCHARGE_RACKS, PROHIBITED_SLOT_INDICES
    # establecer D_racks a nivel de módulo
    globals()['D_racks'] = D_racks
    PROHIBITED_SLOT_INDICES = list(range(4))
    try:
        discharge_racks = sorted(set(slot_to_rack[s] for s in PROHIBITED_SLOT_INDICES if s < len(slot_to_rack)))
    except Exception:
        discharge_racks = []
    DISCHARGE_RACKS = discharge_racks
    if hasattr(VU, '__len__'):
        VU_map = {i+1: float(VU[i]) for i in range(len(VU))}
    elif isinstance(VU, dict):
        VU_map = {int(k): float(v) for k, v in VU.items()}
    else:
        VU_map = {}

    # Para cada solución de slotting, ejecuta el NSGA2 de picking completo
    for idx, slot_assignment in enumerate(slot_assignments):
        # Inicializar población
        population = initialize_population(
            orders, [slot_assignment], Vm, VU, slot_to_rack, START_RACK, pop_size, D_racks
        )
        # Evolución NSGA-II
        for gen in range(n_gen):
            # Evaluar población (usar orden correcto de argumentos)
            population_eval_full = [
                evaluate_individual(ind['genome'], [slot_assignment], slot_to_rack, Vc, START_RACK, orders, Vm)
                for ind in population
            ]
            population_eval_triples = [
                (d, most_demanded_sku_distance(ind['genome'], [slot_assignment], slot_to_rack, orders, start_rack=START_RACK, top_k=5), penalized)
                for (d, sku_dist, penalized, rutas, aug), ind in zip(population_eval_full, population)
            ]
            # Elitismo y reproducción
            offspring = []
            while len(offspring) < pop_size:
                p1, p2 = random.sample(population, 2)
                if p1['cluster_idx'] == p2['cluster_idx']:
                    c1 = order_crossover(p1, p2)
                    c2 = order_crossover(p2, p1)
                else:
                    c1 = swap_mutation(p1)
                    c2 = swap_mutation(p2)
                offspring.append(swap_mutation(c1))
                offspring.append(swap_mutation(c2))
            offspring = offspring[:pop_size]
            # Evaluar hijos
            offspring_eval_full = [
                evaluate_individual(ind['genome'], [slot_assignment], slot_to_rack, Vc, START_RACK, orders, Vm)
                for ind in offspring
            ]
            offspring_eval_triples = [
                (d, most_demanded_sku_distance(ind['genome'], [slot_assignment], slot_to_rack, orders, start_rack=START_RACK, top_k=5), penalized)
                for (d, sku_dist, penalized, rutas, aug), ind in zip(offspring_eval_full, offspring)
            ]
            # Selección elitista
            combined = population + offspring
            objectives = [ind['objectives'] if 'objectives' in ind else (0,0,False) for ind in combined]
            for i, ind in enumerate(combined):
                if 'objectives' not in ind:
                    if i < len(population):
                        ind['objectives'] = population_eval_triples[i]
                    else:
                        ind['objectives'] = offspring_eval_triples[i-len(population)]
            objectives = [ind['objectives'] for ind in combined]
            fronts = fast_non_dominated_sort(objectives)
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= pop_size:
                    new_population.extend([combined[i] for i in front])
                else:
                    distances = crowding_distance(front, objectives)
                    sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
                    slots_remaining = pop_size - len(new_population)
                    new_population.extend([combined[i] for i in sorted_front[:slots_remaining]])
                    break
            population = new_population
        # Evaluar población final
        population_eval_full = [
            evaluate_individual(ind['genome'], [slot_assignment], slot_to_rack, Vc, START_RACK, orders, Vm)
            for ind in population
        ]
        population_eval_triples = [
            (d, most_demanded_sku_distance(ind['genome'], [slot_assignment], slot_to_rack, orders, start_rack=START_RACK, top_k=5), penalized)
            for (d, sku_dist, penalized, rutas, aug), ind in zip(population_eval_full, population)
        ]
        pf = pareto_front(population_eval_triples)
        # Gráfica de Pareto: todos los puntos en azul, solo la mejor solución resaltada con estrella
        fig, ax = plt.subplots(figsize=(7,5))
        xs = [dist for _, dist, sku_dist in pf]
        ys = [sku_dist for _, dist, sku_dist in pf]
        ax.scatter(xs, ys, color='tab:blue', alpha=0.7)
        # Mejor solución (menor distancia total)
        if pf:
            best_idx = min(pf, key=lambda x: x[1])[0]
            best_dist = population_eval_triples[best_idx][0]
            best_sku_dist = population_eval_triples[best_idx][1]
            ax.scatter([best_dist], [best_sku_dist], color='gold', marker='*', s=200, edgecolor='black', label='Mejor')
        ax.set_xlabel('Distancia Total')
        ax.set_ylabel('Distancia a SKUs más demandados')
        ax.set_title(f'Pareto Picking - Solución Slotting {idx+1}')
        ax.grid(True)
        # Rutas por pedido para el mejor individuo (menor distancia total)
        best_idx = min(pf, key=lambda x: x[1])[0] if pf else 0
        rutas_best = population_eval_full[best_idx][3]
        resultados.append({
            'pareto_front': pf,
            'fig': fig,
            'population_eval_triples': population_eval_triples,
            'rutas_best': rutas_best,
            'distancia_total': population_eval_triples[best_idx][0],
            'sku_distancia': population_eval_triples[best_idx][1],
            'penalizado': population_eval_triples[best_idx][2],
        })
    return resultados
