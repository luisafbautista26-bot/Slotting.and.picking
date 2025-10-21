import numpy as np
import picking_solver

# Datos sintéticos pequeños
# 4 SKUs, 6 slots, 3 racks
D = np.array([
    [1,0,2,0],  # pedido 1
    [0,1,0,1],  # pedido 2
])
VU = np.array([0.5, 0.5, 0.5, 0.5])  # vol unitario por sku (index 0 -> sku 1)

# Sr: 6 slots x 3 racks
Sr = np.array([
    [1,0,0],
    [1,0,0],
    [0,1,0],
    [0,1,0],
    [0,0,1],
    [0,0,1],
])

# Distancias entre racks (3x3)
D_racks = np.array([
    [0.0, 1.0, 2.0],
    [1.0, 0.0, 1.5],
    [2.0, 1.5, 0.0]
])

# Slot assignments: 2 soluciones (arrays length 6) — usar 1-based SKU ids
slot1 = np.array([1,2,3,4,1,2])
slot2 = np.array([2,1,4,3,2,1])
slot_assignments = [slot1, slot2]

print('Running nsga2_picking_streamlit with small pop and gen...')
res = picking_solver.nsga2_picking_streamlit(slot_assignments=slot_assignments, D=D, VU=VU, Sr=Sr, D_racks=D_racks, pop_size=6, n_gen=2)

print('Done. Results summary:')
for i, r in enumerate(res):
    print(f"Solution {i+1}: distancia_total={r['distancia_total']}, penalizado={r['penalizado']}, rutas_best={r['rutas_best']}")
