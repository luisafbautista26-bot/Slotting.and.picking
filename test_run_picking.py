import numpy as np
from picking_solver import nsga2_picking_streamlit

# Parámetros de prueba
NUM_SLOTS = 12
NUM_RACKS = 6
NUM_SKUS = 4  # SKUs indexados 0..3

# Crear dos asignaciones de slots (dos soluciones de slotting)
slot_assign1 = np.array([1,2,3,1,2,3,1,2,3,0,0,0])
slot_assign2 = np.array([2,1,3,2,1,3,2,1,3,0,0,0])
slot_assignments = [slot_assign1, slot_assign2]

# Pedidos: 3 pedidos, matriz D (3 x NUM_SKUS)
D = np.array([
    [0,1,0,0],  # pedido 0: 1 unidad del SKU 1
    [0,0,2,0],  # pedido 1: 2 unidades del SKU 2
    [0,1,1,0],  # pedido 2: 1 unidad SKU1 y 1 unidad SKU2
])

# Volumen unitario por SKU (indexable por sku)
VU = np.array([0.0, 1.0, 1.0, 1.0])

# Sr: matriz slots x racks (one-hot)
Sr = np.zeros((NUM_SLOTS, NUM_RACKS), dtype=int)
for s in range(NUM_SLOTS):
    rack = s % NUM_RACKS
    Sr[s, rack] = 1

# Matriz de distancias entre racks (simétrica)
coords = np.array([[i, (i*2)%NUM_RACKS] for i in range(NUM_RACKS)], dtype=float)
D_racks = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=2)

print('Running nsga2_picking_streamlit test...')
results = nsga2_picking_streamlit(slot_assignments, D, VU, Sr, D_racks, pop_size=6, n_gen=3)

for i, res in enumerate(results):
    print(f"\n--- Result for slot_assignment {i} ---")
    print('Distancia total (mejor):', res['distancia_total'])
    print('Penalizado:', res['penalizado'])
    print('Rutas mejor individuo:')
    for j, ruta in enumerate(res['rutas_best']):
        print(f'  Pedido {j}:', ' -> '.join(map(str, ruta)))

    # Verificación: ninguna ruta debe tener 0 -> discharge point como primer movimiento
    DISCHARGE_POINTS = [1,2,3]
    bad = False
    for ruta in res['rutas_best']:
        if len(ruta) >= 2 and ruta[1] in DISCHARGE_POINTS:
            bad = True
            print('  ERROR: ruta comienza en punto de descarga:', ruta)
    if not bad:
        print('  OK: ninguna ruta comienza en punto de descarga')

print('\nTest finished')
