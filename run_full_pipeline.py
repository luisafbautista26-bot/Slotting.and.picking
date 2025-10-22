import pandas as pd
import numpy as np
import traceback
from nsga2_solver import nsga2
from picking_solver import nsga2_picking_streamlit

excel_path = 'pedidos2.0 (2).xlsx'
print('Loading', excel_path)
excel = pd.ExcelFile(excel_path)
print('Sheets:', excel.sheet_names)

# load data
VU = None
D = None
Sr = None
D_racks = None
if 'VU' in excel.sheet_names:
    VU_df = excel.parse('VU', header=None)
    VU = VU_df.values.flatten()
if 'D' in excel.sheet_names:
    D = excel.parse('D').values
else:
    raise SystemExit('Sheet D required')
if 'Sr' in excel.sheet_names:
    Sr = excel.parse('Sr').values
else:
    print('Warning: Sr missing, inferring')
if 'D_racks' in excel.sheet_names:
    D_racks = excel.parse('D_racks', header=None).values

# sanitize VU
if VU is None:
    VU = np.ones(D.shape[1])
else:
    VU = np.array(VU, dtype=float)
    VU = np.nan_to_num(VU, nan=1.0)

# sanitize Sr
if Sr is None:
    NUM_SLOTS = 12
    NUM_RACKS = D_racks.shape[0] if D_racks is not None else 6
    Sr = np.zeros((NUM_SLOTS, NUM_RACKS), dtype=int)
    for s in range(NUM_SLOTS):
        Sr[s, s % NUM_RACKS] = 1

slot_assignment = np.argmax(Sr, axis=1)
NUM_SLOTS = slot_assignment.shape[0]
slot_to_rack = np.argmax(Sr, axis=1).tolist()

# sanitize D_racks
if D_racks is None:
    NUM_RACKS = max(1, int(np.max(slot_to_rack))+1)
    coords = np.array([[i, (i*2)%NUM_RACKS] for i in range(NUM_RACKS)], dtype=float)
    D_racks = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=2)
else:
    D_racks = np.array(D_racks, dtype=float)
    if D_racks.shape[0] != D_racks.shape[1]:
        m = min(D_racks.shape[0], D_racks.shape[1])
        D_racks = D_racks[:m, :m]
    expected_racks = int(np.max(slot_to_rack)) + 1
    if D_racks.shape[0] < expected_racks:
        pad = expected_racks - D_racks.shape[0]
        fill_val = np.nanmax(D_racks[np.isfinite(D_racks)]) if np.isfinite(D_racks).any() else 1e5
        big = fill_val * 10
        D_new = np.full((expected_racks, expected_racks), big, dtype=float)
        D_new[:D_racks.shape[0], :D_racks.shape[1]] = D_racks
        D_racks = D_new
    D_racks = np.nan_to_num(D_racks, nan=(np.nanmax(D_racks[np.isfinite(D_racks)]) if np.isfinite(D_racks).any() else 1e5)*10)
    D_racks = (D_racks + D_racks.T) / 2.0
    np.fill_diagonal(D_racks, 0.0)

print('Running slotting NSGA-II (defaults)...')
try:
    pareto_solutions, pareto_fitness = nsga2(
        pop_size=30,
        generations=50,
        cx_rate=0.9,
        pm_swap=0.3,
        seed=42,
        NUM_SLOTS=NUM_SLOTS,
        PROHIBITED_SLOTS={0,1,2,3},
        NUM_SKUS=D.shape[1],
        D=D,
        VU=VU,
        Vm=3,
        rack_assignment=slot_to_rack,
        D_racks=D_racks
    )
    print('Slotting produced', len(pareto_solutions), 'pareto solutions')
except Exception as e:
    print('Slotting failed:', e)
    print(traceback.format_exc())
    raise

print('Running picking on all slotting solutions...')
results = nsga2_picking_streamlit(
    slot_assignments=pareto_solutions,
    D=D,
    VU=VU,
    Sr=Sr,
    D_racks=D_racks,
    pop_size=20,
    n_gen=10
)

for i, r in enumerate(results):
    print('\nSlotting solution', i+1)
    print(' Pareto points (count):', len(r['pareto_front']))
    print(' Best distancia_total:', r['distancia_total'])
    print(' Penalizado:', r['penalizado'])
    print(' Rutas:')
    for j, ruta in enumerate(r['rutas_best']):
        print('  Pedido', j+1, ':', ' -> '.join(map(str, ruta)))

print('\nDone')

# Guardar resultados en CSV para inspección
import csv
summary_file = 'picking_summary.csv'
with open(summary_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['slotting_idx', 'pareto_count', 'distancia_total', 'sku_distancia', 'penalizado'])
    for i, r in enumerate(results):
        writer.writerow([i+1, len(r['pareto_front']), r['distancia_total'], r['sku_distancia'], r['penalizado']])

for i, r in enumerate(results):
    routes_file = f'picking_routes_slotting_{i+1}.csv'
    with open(routes_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pedido_idx', 'ruta'])
        for j, ruta in enumerate(r['rutas_best']):
            writer.writerow([j+1, ' -> '.join(map(str, ruta))])

print(f"CSV summary guardado en: {summary_file}")
print("CSV de rutas por slotting: picking_routes_slotting_#.csv")
