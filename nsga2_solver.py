import numpy as np
import random
import math
from collections import defaultdict

# =============================
# === Funciones NSGA-II modularizadas ===
# =============================

def calcular_required_slots(D, VU, Vm):
    demand_total = D.sum(axis=0)
    required_slots = {}
    for sku_idx in range(D.shape[1]):
        sku_id = sku_idx  # Corregido: los índices deben coincidir con VU y D
        required_slots[sku_id] = math.ceil(demand_total[sku_idx] * VU[sku_id] / Vm)
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
            sku_id = sku_idx + 1
            demanda = D[pedido_idx, sku_idx]
            if demanda > 0:
                sku_slots = [slot for slot, sku_val in enumerate(ind) if sku_val == sku_id]
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

def crossover_uniform(p1, p2, px, repair, NUM_SLOTS, PROHIBITED_SLOTS, required_slots):
    n = len(p1)
    child = np.zeros(n, dtype=int)
    mask = np.random.rand(n) < px
    child[mask] = p1[mask]
    child[~mask] = p2[~mask]
    return repair(child, NUM_SLOTS, PROHIBITED_SLOTS, required_slots)

def mutate_swap(ind, pm, repair, NUM_SLOTS, PROHIBITED_SLOTS, required_slots):
    ind = ind.copy()
    if random.random() < pm:
        valid_slots = [s for s in range(NUM_SLOTS) if s not in PROHIBITED_SLOTS]
        if len(valid_slots) >= 2:
            i, j = random.sample(valid_slots, 2)
            ind[i], ind[j] = ind[j], ind[i]
    return repair(ind, NUM_SLOTS, PROHIBITED_SLOTS, required_slots)

def repair(ind, NUM_SLOTS, PROHIBITED_SLOTS, required_slots):
    ind = ind.copy()
    current_counts = defaultdict(int)
    for sku in ind:
        if sku > 0:
            current_counts[sku] += 1
    for sku in list(current_counts.keys()):
        required = required_slots.get(sku, 0)
        if current_counts[sku] > required:
            sku_slots = [i for i, val in enumerate(ind) if val == sku and i not in PROHIBITED_SLOTS]
            excess = current_counts[sku] - required
            for i in range(excess):
                if i < len(sku_slots):
                    ind[sku_slots[i]] = 0
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
    required_slots = calcular_required_slots(D, VU, Vm)
    all_assignments = []
    demand_total = D.sum(axis=0)
    for sku_idx in range(NUM_SKUS):
        sku_id = sku_idx  # Corregido: los índices deben coincidir con VU y D
        count = math.ceil(demand_total[sku_idx] * VU[sku_id] / Vm)
        all_assignments.extend([sku_id] * count)
    usable_slots = NUM_SLOTS - len(PROHIBITED_SLOTS)
    if len(all_assignments) < usable_slots:
        all_assignments.extend([0] * (usable_slots - len(all_assignments)))
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
