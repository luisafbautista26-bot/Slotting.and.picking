# picking_solver.py
import numpy as np
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt

__all__ = ["nsga2_picking_streamlit"]

# --- Funciones auxiliares completas para el algoritmo de picking ---
def capacidad_slot(slot, slot_assignment_row, Vm, VU):
    sku = slot_assignment_row[slot]
    if sku == 0: return 0
    return int(Vm[slot] // VU[sku])

def route_for_order(order, slot_assignment_row, Vm, VU, slot_to_rack, start_rack, D_racks):
    DISCHARGE_POINTS = [1, 2, 3]
    Vm_local = Vm.copy()
    genome = [0]
    indices = np.where(order > 0)[0]
    cantidades = order[order > 0]
    remaining = {sku: qty for sku, qty in zip(indices, cantidades)}
    racks_needed = set()
    for sku, demanda in remaining.items():
        for slot, s in enumerate(slot_assignment_row):
            if s == sku and slot not in DISCHARGE_POINTS:
                capacidad = int(Vm_local[slot] // VU[sku])
                if capacidad > 0:
                    recoger = min(demanda, capacidad)
                    demanda -= recoger
                    Vm_local[slot] -= recoger * VU[sku]
                    racks_needed.add(slot_to_rack[slot])
                    if demanda <= 0:
                        break
    # Evitar que la primera parada sea un punto de descarga: filtrar racks que sean discharge
    orig_racks = list(racks_needed)
    racks_needed = [r for r in orig_racks if r not in DISCHARGE_POINTS]
    # Si tras filtrar no queda ningún rack (todos eran puntos de descarga), devolvemos una subruta vacía
    # para evitar el recorrido 0 -> discharge (se penalizará si no hay pickups reales)
    if not racks_needed:
        return [0, 0]
    route = []
    current = start_rack
    while racks_needed:
        next_rack = min(racks_needed, key=lambda r: D_racks[current, r])
        route.append(next_rack)
        current = next_rack
        racks_needed.remove(next_rack)
    genome += route + [0]
    return genome

def build_genome(orders, slot_assignment_row, Vm, VU, slot_to_rack, start_rack, cluster_idx, D_racks):
    genome = [cluster_idx, 0]
    for order in orders:
        subroute = route_for_order(order, slot_assignment_row, Vm, VU, slot_to_rack, start_rack, D_racks)
        genome += subroute[1:]  # skip extra zero from subroute
    return genome

def initialize_population(orders, slot_assignments, Vm, VU, slot_to_rack, start_rack, pop_size, D_racks):
    population = []
    num_clusters = len(slot_assignments)
    num_orders = len(orders)
    for cluster_idx, slot_assignment_row in enumerate(slot_assignments):
        genome = build_genome(orders, slot_assignment_row, Vm, VU, slot_to_rack, start_rack, cluster_idx, D_racks)
        population.append({'genome': genome, 'cluster_idx': cluster_idx})
    while len(population) < pop_size:
        cluster_idx = random.randint(0, num_clusters-1)
        slot_assignment_row = slot_assignments[cluster_idx]
        genome = [cluster_idx, 0]
        for order in orders:
            Vm_local = Vm.copy()
            indices = np.where(order > 0)[0]
            cantidades = order[order > 0]
            remaining = {sku: qty for sku, qty in zip(indices, cantidades)}
            racks_needed = set()
            for sku, demanda in remaining.items():
                for slot, s in enumerate(slot_assignment_row):
                    if s == sku and slot not in [1,2,3]:
                        capacidad = int(Vm_local[slot] // VU[sku])
                        if capacidad > 0:
                            recoger = min(demanda, capacidad)
                            demanda -= recoger
                            Vm_local[slot] -= recoger * VU[sku]
                            racks_needed.add(slot_to_rack[slot])
                            if demanda <= 0:
                                break
            racks_needed = list(racks_needed)
            random.shuffle(racks_needed)
            genome += racks_needed + [0]
        # Asegurar que el número de pedidos (subrutas) sea igual al de la matriz de pedidos
        ceros = [i for i, v in enumerate(genome) if v == 0]
        if len(ceros) < num_orders + 2:
            genome += [0] * (num_orders + 2 - len(ceros))
        elif len(ceros) > num_orders + 2:
            extras = len(ceros) - (num_orders + 2)
            idxs = [i for i, v in enumerate(genome) if v == 0][1:]
            for _ in range(extras):
                genome.pop(idxs.pop())
        population.append({'genome': genome, 'cluster_idx': cluster_idx})
    return population

def insert_discharge_points_and_boxes(genome, slot_assignments, D_racks, slot_to_rack, VU,
                                      box_volume_max=1, start_rack=0, orders=None):
    DISCHARGE_POINTS = [1, 2, 3]
    cluster_idx = genome[0]
    slot_assignment_row = slot_assignments[cluster_idx]
    new_genome = [cluster_idx, 0]
    i = 2
    order_idx = 0
    current_rack = start_rack
    box_vol = 0.0
    orders = np.array(orders)
    order_demands = [dict((sku, int(qty)) for sku, qty in enumerate(order) if qty > 0) for order in orders]
    num_orders = len(orders)

    # Flag para saber si se recogió algo desde el último punto de descarga
    pickups_since_last = False

    while i < len(genome):
        rack = genome[i]
        if rack == 0:
            # Sólo insertar un punto de descarga si entre el último discharge (o inicio)
            # se recogió al menos un item
            if pickups_since_last:
                if new_genome[-1] not in DISCHARGE_POINTS:
                    last_rack = current_rack
                    nearest_discharge = min(DISCHARGE_POINTS, key=lambda dp: D_racks[last_rack, dp])
                    new_genome.append(nearest_discharge)
                    current_rack = nearest_discharge
                    box_vol = 0.0
            # Siempre añadimos el separador de pedido (0)
            new_genome.append(0)
            order_idx += 1
            current_rack = start_rack
            box_vol = 0.0
            pickups_since_last = False
            i += 1
            continue
        new_genome.append(rack)
        current_rack = rack
        if order_idx < num_orders:
            demand = order_demands[order_idx]
            picked_any = False
            for slot in range(len(slot_assignment_row)):
                sku = slot_assignment_row[slot]
                if slot_to_rack[slot] == rack and sku in demand and slot not in DISCHARGE_POINTS and demand[sku] > 0:
                    unit_vol = VU[sku]
                    qty_to_pick = demand[sku]
                    if qty_to_pick > 0:
                        picked_any = True
                    for _ in range(qty_to_pick):
                        if box_vol + unit_vol > box_volume_max:
                            nearest_discharge = min(DISCHARGE_POINTS, key=lambda dp: D_racks[current_rack, dp])
                            new_genome.append(nearest_discharge)
                            current_rack = nearest_discharge
                            box_vol = 0.0
                        box_vol += unit_vol
                    demand[sku] = 0
            if picked_any:
                pickups_since_last = True
        i += 1
    filtered_genome = [new_genome[0], new_genome[1]]
    for i in range(2, len(new_genome)):
        if new_genome[i] in DISCHARGE_POINTS and new_genome[i-1] == new_genome[i]:
            continue
        filtered_genome.append(new_genome[i])
    return filtered_genome

def most_demanded_sku_distance(genome, slot_assignments, D_racks, orders, slot_to_rack, start_rack=0, top_k=5):
    cluster_idx = genome[0]
    slot_assignment_row = slot_assignments[cluster_idx]
    sku_demands = np.sum(orders, axis=0)
    nonzero_skus = np.where(sku_demands > 0)[0]
    top_skus = nonzero_skus[np.argsort(sku_demands[nonzero_skus])[::-1][:top_k]]
    total_distance = 0.0
    for sku in top_skus:
        slots = np.where(slot_assignment_row == sku)[0]
        for slot in slots:
            rack = slot_to_rack[slot]
            total_distance += D_racks[start_rack, rack]
    return total_distance

def evaluate_individual(genome, slot_assignments, D_racks, slot_to_rack, VU, Vm,
                       box_volume_max=1, start_rack=0, orders=None):
    cluster_idx = genome[0]
    slot_assignment_row = slot_assignments[cluster_idx]
    genome_with_discharges = insert_discharge_points_and_boxes(
        genome, slot_assignments, D_racks, slot_to_rack, VU,
        box_volume_max=1, start_rack=start_rack, orders=orders
    )
    total_distance = 0.0
    current_rack = start_rack
    order_idx = 0
    penalized = False
    i = 2
    ruta_actual = [start_rack]
    rutas_por_pedido = []
    orders = np.array(orders)
    order_demands = [dict((sku, int(qty)) for sku, qty in enumerate(order) if qty > 0) for order in orders]
    num_orders = len(orders)
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
    Vm = np.full(NUM_SLOTS, 3)
    START_RACK = 0
    DISCHARGE_POINTS = [1, 2, 3]
    orders = np.array(D)
    slot_to_rack = np.argmax(Sr, axis=1).tolist()
    Vc = 1

    # Para cada solución de slotting, ejecuta el NSGA2 de picking completo
    for idx, slot_assignment in enumerate(slot_assignments):
        # Inicializar población
        population = initialize_population(
            orders, [slot_assignment], Vm, VU, slot_to_rack, START_RACK, pop_size, D_racks
        )
        # Evolución NSGA-II
        for gen in range(n_gen):
            population_eval_full = [
                evaluate_individual(ind['genome'], [slot_assignment], D_racks, slot_to_rack, VU, Vm, box_volume_max=Vc, start_rack=START_RACK, orders=orders)
                for ind in population
            ]
            population_eval_triples = [
                (d, most_demanded_sku_distance(ind['genome'], [slot_assignment], D_racks, orders, slot_to_rack, start_rack=START_RACK, top_k=5), p)
                for (d, b, p, rutas), ind in zip(population_eval_full, population)
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
                evaluate_individual(ind['genome'], [slot_assignment], D_racks, slot_to_rack, VU, Vm, box_volume_max=Vc, start_rack=START_RACK, orders=orders)
                for ind in offspring
            ]
            offspring_eval_triples = [
                (d, most_demanded_sku_distance(ind['genome'], [slot_assignment], D_racks, orders, slot_to_rack, start_rack=START_RACK, top_k=5), p)
                for (d, b, p, rutas), ind in zip(offspring_eval_full, offspring)
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
            evaluate_individual(ind['genome'], [slot_assignment], D_racks, slot_to_rack, VU, Vm, box_volume_max=Vc, start_rack=START_RACK, orders=orders)
            for ind in population
        ]
        population_eval_triples = [
            (d, most_demanded_sku_distance(ind['genome'], [slot_assignment], D_racks, orders, slot_to_rack, start_rack=START_RACK, top_k=5), p)
            for (d, b, p, rutas), ind in zip(population_eval_full, population)
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
