# picking_solver.py
import numpy as np
import random
import math
from collections import defaultdict

def nsga2_picking(slot_assignments, D, VU, Sr, D_racks, pop_size=10, n_gen=5):
	"""
	slot_assignments: lista de arrays (cada uno es una solución de slotting)
	D: matriz de pedidos (num_pedidos x num_skus)
	VU: vector de volumen unitario por SKU (dict o array)
	Sr: matriz de slots x racks (1 si el slot pertenece al rack)
	D_racks: matriz de distancias entre racks
	"""
	resultados = []
	for idx, slot_assignment in enumerate(slot_assignments):
		# Para cada solución de slotting, ejecuta el NSGA2 de picking
		# Aquí se asume que cada slot_assignment es un array de asignación de SKU a slot
		# --- Lógica simplificada de picking NSGA2 ---
		# Puedes reemplazar esto por tu lógica real de picking
		rutas, distancia_total, sku_distancia = resolver_picking(slot_assignment, D, Sr, D_racks)
		resultados.append({
			'distancia_total': distancia_total,
			'sku_distancia': sku_distancia,
			'rutas': rutas
		})
	return resultados

def resolver_picking(slot_assignment, D, Sr, D_racks):
	"""
	slot_assignment: array de asignación de SKU a slot
	D: matriz de pedidos (num_pedidos x num_skus)
	Sr: matriz de slots x racks
	D_racks: matriz de distancias entre racks
	"""
	num_pedidos, num_skus = D.shape
	rutas = []
	distancia_total = 0.0
	sku_distancia = 0.0
	for pedido_idx in range(num_pedidos):
		skus_pedido = np.where(D[pedido_idx] > 0)[0]
		racks_visitados = set()
		ruta = [0]  # Empieza en rack 0 (suponiendo que es el punto de partida)
		for sku_idx in skus_pedido:
			sku_id = sku_idx + 1
			slots_sku = np.where(slot_assignment == sku_id)[0]
			if len(slots_sku) == 0:
				continue
			slot = slots_sku[0]
			rack = np.where(Sr[slot] == 1)[0][0]
			racks_visitados.add(rack)
		racks_orden = sorted(list(racks_visitados))
		ruta.extend(racks_orden)
		ruta.append(0)  # Regresa al inicio
		rutas.append(ruta)
		# Calcular distancia total de la ruta
		dist = 0.0
		for i in range(len(ruta)-1):
			dist += D_racks[ruta[i], ruta[i+1]]
		distancia_total += dist
		# Calcular distancia de los SKUs demandados
		sku_dist = 0.0
		for sku_idx in skus_pedido:
			sku_id = sku_idx + 1
			slots_sku = np.where(slot_assignment == sku_id)[0]
			if len(slots_sku) == 0:
				continue
			slot = slots_sku[0]
			rack = np.where(Sr[slot] == 1)[0][0]
			sku_dist += D_racks[0, rack]
		sku_distancia += sku_dist
	return rutas, distancia_total, sku_distancia
# Implementación modularizada del NSGA2 de picking, adaptada para integración con Streamlit
# (El código será completado en el siguiente paso)
