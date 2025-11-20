import numpy as np
import random
import math
from collections import defaultdict

# =============================
# === Defaults / seguridad ===
# =============================
# Garantizar que los puntos prohibidos siempre incluyan los puntos de descarga
# 0,1,2,3 (0 es punto de inicio). Si el usuario carga estas variables desde
# un excel/otro script, estas asignaciones serán respetadas; si no, usamos
# valores por defecto seguros.
if 'PROHIBITED_SLOTS' not in globals():
    PROHIBITED_SLOTS = {0, 1, 2, 3}

# Capacidad por slot por defecto (si el usuario no lo proporciona)
if 'Vm' not in globals():
    Vm = 3

# =============================
# === Funciones NSGA-II modularizadas ===
# =============================

def calcular_required_slots(D, VU, Vm):
    demand_total = D.sum(axis=0)
    required_slots = {}
    for sku_idx in range(D.shape[1]):
        # Usamos convención 1-based para SKU ids: sku_id = sku_idx + 1
        sku_id = sku_idx + 1
        # VU puede ser array-like 0-based (VU[0] -> sku 1) o dict con keys 1-based
        if isinstance(VU, dict):
            vu_val = float(VU.get(sku_id, 0.0))
        else:
            vu_val = float(VU[sku_idx])
        
        # Manejar NaN y valores inválidos
        demand_val = demand_total[sku_idx]
        if np.isnan(demand_val) or np.isnan(vu_val) or vu_val <= 0 or Vm <= 0:
            required_slots[sku_id] = 0
        else:
            try:
                required_slots[sku_id] = math.ceil(demand_val * vu_val / Vm)
            except (ValueError, OverflowError):
                required_slots[sku_id] = 0
    return required_slots

def get_rack_for_slot(slot_idx, rack_assignment):
    return rack_assignment[slot_idx]

def get_distance_between_racks(rack1, rack2, D_racks):
    return D_racks[rack1, rack2]

def init_individual(all_assignments, NUM_SLOTS, PROHIBITED_SLOTS):
    genes = all_assignments.copy()
    random.shuffle(genes)
    ind = np.zeros(NUM_SLOTS, dtype=int)
    g_idx = 0
    for slot in range(NUM_SLOTS):
        if slot in PROHIBITED_SLOTS:
            ind[slot] = 0
        else:
            ind[slot] = genes[g_idx]
            g_idx += 1
    return ind

def fitness(ind, required_slots, NUM_SKUS, D, VU, Vm, rack_assignment, D_racks, PROHIBITED_SLOTS):
    f1, f2 = 0, 0
    penalty = 0
    slot_counts = defaultdict(int)
    for sku in ind:
        if sku > 0:
            slot_counts[sku] += 1
    for sku, required in required_slots.items():
        if slot_counts[sku] < required:
            penalty += (required - slot_counts[sku]) * 1000
    sku_racks = defaultdict(list)
    for slot, sku in enumerate(ind):
        if sku > 0:
            rack = get_rack_for_slot(slot, rack_assignment)
            sku_racks[sku].append(rack)
    for sku, racks in sku_racks.items():
        if len(racks) > 1:
            total_dist = 0
            count = 0
            for i in range(len(racks)):
                for j in range(i + 1, len(racks)):
                    total_dist += get_distance_between_racks(racks[i], racks[j], D_racks)
                    count += 1
            if count > 0:
                f1 += total_dist / count
    num_pedidos = D.shape[0]
    for pedido_idx in range(num_pedidos):
        for sku_idx in range(NUM_SKUS):
            # sku_id en la convención 1-based
            sku_id = sku_idx + 1
            demanda = D[pedido_idx, sku_idx]
            if demanda > 0:
                sku_slots = [slot for slot, sku_val in enumerate(ind) if int(sku_val) == sku_id]
                if not sku_slots:
                    penalty += demanda * 1000
                else:
                    min_distance = float('inf')
                    for slot in sku_slots:
                        rack = get_rack_for_slot(slot, rack_assignment)
                        distance = get_distance_between_racks(0, rack, D_racks)
                        min_distance = min(min_distance, distance)
                    f2 += demanda * min_distance
    f1 += penalty
    f2 += penalty
    return (f1, f2)

def dominates(fitness1, fitness2):
    better_or_equal = (fitness1[0] <= fitness2[0] and fitness1[1] <= fitness2[1])
    strictly_better = (fitness1[0] < fitness2[0] or fitness1[1] < fitness2[1])
    return better_or_equal and strictly_better

def crossover_uniform(p1, p2, px, repair, NUM_SLOTS=None, PROHIBITED_SLOTS=None, required_slots=None):
    n = len(p1)
    child = np.zeros(n, dtype=int)
    mask = np.random.rand(n) < px
    child[mask] = p1[mask]
    child[~mask] = p2[~mask]
    # Compatibilidad: intentar llamar a repair con la firma completa;
    # si la función repair original solo acepta (ind), caer en el fallback.
    try:
        return repair(child, NUM_SLOTS, PROHIBITED_SLOTS, required_slots)
    except TypeError:
        return repair(child)

def mutate_swap(ind, pm, repair, NUM_SLOTS=None, PROHIBITED_SLOTS=None, required_slots=None):
    ind = ind.copy()
    if random.random() < pm:
        valid_slots = [s for s in range(NUM_SLOTS) if s not in PROHIBITED_SLOTS]
        if len(valid_slots) >= 2:
            i, j = random.sample(valid_slots, 2)
            ind[i], ind[j] = ind[j], ind[i]
    try:
        return repair(ind, NUM_SLOTS, PROHIBITED_SLOTS, required_slots)
    except TypeError:
        return repair(ind)

def repair(ind, NUM_SLOTS=None, PROHIBITED_SLOTS=None, required_slots=None):
    """Repara un individuo para cumplir con los slots requeridos.

    Firma flexible: puede llamarse como repair(ind) (usa globals) o
    repair(ind, NUM_SLOTS, PROHIBITED_SLOTS, required_slots).
    """
    ind = ind.copy()

    # Resolver parámetros: si no se pasaron, leer desde globals
    if NUM_SLOTS is None:
        NUM_SLOTS = globals().get('NUM_SLOTS', None)
    if PROHIBITED_SLOTS is None:
        PROHIBITED_SLOTS = globals().get('PROHIBITED_SLOTS', [])
    if required_slots is None:
        required_slots = globals().get('required_slots', None)

    # Si required_slots no está disponible, nada que reparar
    if required_slots is None:
        return ind

    # Contar slots actuales por SKU
    current_counts = defaultdict(int)
    for sku in ind:
        if sku > 0:
            current_counts[sku] += 1

    # Quitar excedentes (determinístico: los primeros encontrados)
    for sku in list(current_counts.keys()):
        required = required_slots.get(sku, 0)
        if current_counts[sku] > required:
            sku_slots = [i for i, val in enumerate(ind) if val == sku and i not in PROHIBITED_SLOTS]
            excess = current_counts[sku] - required
            for i in range(excess):
                if i < len(sku_slots):
                    ind[sku_slots[i]] = 0

    # Agregar faltantes
    for sku, required in required_slots.items():
        current = sum(1 for val in ind if val == sku)
        if current < required:
            empty_slots = [i for i, val in enumerate(ind) if val == 0 and i not in PROHIBITED_SLOTS]
            needed = required - current
            for i in range(min(needed, len(empty_slots))):
                ind[empty_slots[i]] = sku

    return ind

def non_dominated_sort(population, fitness_func):
    population_size = len(population)
    fitness_values = [fitness_func(ind) for ind in population]
    S = [[] for _ in range(population_size)]
    n = [0] * population_size
    rank = [0] * population_size
    fronts = [[]]
    for i in range(population_size):
        for j in range(population_size):
            if i != j:
                if dominates(fitness_values[i], fitness_values[j]):
                    S[i].append(j)
                elif dominates(fitness_values[j], fitness_values[i]):
                    n[i] += 1
        if n[i] == 0:
            rank[i] = 0
            fronts[0].append(i)
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
        if next_front:
            fronts.append(next_front)
        else:
            break
    return fronts, fitness_values

def crowding_distance_assignment(front, fitness_values):
    size = len(front)
    distances = [0.0] * size
    if size == 0:
        return distances
    for m in range(2):
        sorted_indices = sorted(range(size), key=lambda i: fitness_values[front[i]][m])
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        f_min = fitness_values[front[sorted_indices[0]]][m]
        f_max = fitness_values[front[sorted_indices[-1]]][m]
        if abs(f_max - f_min) > 1e-10:
            for i in range(1, size - 1):
                idx = sorted_indices[i]
                prev_fitness = fitness_values[front[sorted_indices[i-1]]][m]
                next_fitness = fitness_values[front[sorted_indices[i+1]]][m]
                distances[idx] += (next_fitness - prev_fitness) / (f_max - f_min)
    return distances

def nsga2(
    pop_size, generations, cx_rate, pm_swap, seed,
    NUM_SLOTS, PROHIBITED_SLOTS, NUM_SKUS, D, VU, Vm, rack_assignment, D_racks
):
    random.seed(seed)
    np.random.seed(seed)
    # Calcular slots requeridos por SKU usando la función modular
    required_slots = calcular_required_slots(D, VU, Vm)

    # Demanda total de cada SKU (suma vertical de columnas)
    demand_total = D.sum(axis=0)

    # Mostrar información útil
    print("Demanda total:", demand_total)
    print("Slots requeridos por SKU:", required_slots)

    # Lista completa de asignaciones necesarias
    all_assignments = []
    for sku, count in required_slots.items():
        all_assignments.extend([sku] * count)

    # Si hay menos asignaciones que slots disponibles → rellenar con vacíos
    usable_slots = NUM_SLOTS - len(PROHIBITED_SLOTS)
    if len(all_assignments) < usable_slots:
        all_assignments.extend([0] * (usable_slots - len(all_assignments)))  # 0 = vacío
    def fitness_func(ind):
        return fitness(ind, required_slots, NUM_SKUS, D, VU, Vm, rack_assignment, D_racks, PROHIBITED_SLOTS)
    def repair_func(ind, NUM_SLOTS, PROHIBITED_SLOTS, required_slots):
        return repair(ind, NUM_SLOTS, PROHIBITED_SLOTS, required_slots)
    pop = [init_individual(all_assignments, NUM_SLOTS, PROHIBITED_SLOTS) for _ in range(pop_size)]
    for g in range(generations):
        off = []
        while len(off) < pop_size:
            p1, p2 = random.sample(pop, 2)
            child = crossover_uniform(p1, p2, cx_rate, repair_func, NUM_SLOTS, PROHIBITED_SLOTS, required_slots)
            child = mutate_swap(child, pm_swap, repair_func, NUM_SLOTS, PROHIBITED_SLOTS, required_slots)
            off.append(child)
        combined_pop = pop + off
        fronts, combined_fitness = non_dominated_sort(combined_pop, fitness_func)
        crowding_distances = []
        for front in fronts:
            if front:
                crowding_distances.append(crowding_distance_assignment(front, combined_fitness))
            else:
                crowding_distances.append([])
        new_pop = []
        remaining = pop_size
        for front_idx, front in enumerate(fronts):
            if len(front) <= remaining:
                new_pop.extend([combined_pop[i] for i in front])
                remaining -= len(front)
            else:
                front_indices = list(zip(front, crowding_distances[front_idx]))
                front_indices.sort(key=lambda x: x[1], reverse=True)
                for i in range(remaining):
                    new_pop.append(combined_pop[front_indices[i][0]])
                break
        pop = new_pop
    final_fitness = [fitness_func(ind) for ind in pop]
    fronts, _ = non_dominated_sort(pop, fitness_func)
    if fronts and fronts[0]:
        pareto_front = [pop[i] for i in fronts[0]]
        pareto_fitness = [final_fitness[i] for i in fronts[0]]
        return pareto_front, pareto_fitness
    else:
        return pop, final_fitness


def nsga2_compat(pop_size, generations, cx_rate, pm_swap, seed, verbose=True):
    """Compatibility wrapper matching the simple signature used in older code.

    It collects required data from globals (D, VU, Vm, NUM_SLOTS, PROHIBITED_SLOTS,
    NUM_SKUS, rack_assignment, D_racks) and calls the explicit `nsga2`.
    """
    # Try to read necessary globals
    G = globals()
    missing = []
    for name in ('D', 'VU', 'Vm', 'NUM_SLOTS', 'PROHIBITED_SLOTS', 'NUM_SKUS', 'rack_assignment', 'D_racks'):
        if name not in G:
            missing.append(name)
    if missing:
        raise RuntimeError(f"nsga2_compat: faltan variables globales necesarias: {missing}")

    D = G['D']
    VU = G['VU']
    Vm = G['Vm']
    NUM_SLOTS = G['NUM_SLOTS']
    PROHIBITED_SLOTS = G['PROHIBITED_SLOTS']
    NUM_SKUS = G['NUM_SKUS']
    rack_assignment = G['rack_assignment']
    D_racks = G['D_racks']

    # Call the explicit nsga2 implementation
    return nsga2(
        pop_size=pop_size,
        generations=generations,
        cx_rate=cx_rate,
        pm_swap=pm_swap,
        seed=seed,
        NUM_SLOTS=NUM_SLOTS,
        PROHIBITED_SLOTS=PROHIBITED_SLOTS,
        NUM_SKUS=NUM_SKUS,
        D=D,
        VU=VU,
        Vm=Vm,
        rack_assignment=rack_assignment,
        D_racks=D_racks,
    )
