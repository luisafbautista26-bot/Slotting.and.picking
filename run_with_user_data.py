import pandas as pd
import numpy as np
from picking_solver import nsga2_picking_streamlit

excel_path = 'pedidos2.0 (2).xlsx'
print('Loading', excel_path)
excel = pd.ExcelFile(excel_path)
print('Sheets found:', excel.sheet_names)

# Intentar cargar las hojas esperadas
VU = None
D = None
Sr = None
D_racks = None

if 'VU' in excel.sheet_names:
    VU = excel.parse('VU', header=None).values.flatten()
else:
    print("Warning: hoja 'VU' no encontrada. Intentaré inferir desde 'D' si es posible.")

if 'D' in excel.sheet_names:
    D = excel.parse('D').values
else:
    print("Error: hoja 'D' (pedidos) no encontrada. No puedo continuar.")
    raise SystemExit(1)

if 'Sr' in excel.sheet_names:
    Sr = excel.parse('Sr').values
else:
    print("Warning: hoja 'Sr' no encontrada. Intentaré inferir Sr usando número de racks 6.")

if 'D_racks' in excel.sheet_names:
    D_racks = excel.parse('D_racks', header=None).values
else:
    print("Warning: hoja 'D_racks' no encontrada. Generando una matriz de distancias Euclidiana entre 6 racks.")
    coords = np.array([[i, (i*2)%6] for i in range(6)], dtype=float)
    D_racks = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=2)

# Inferir VU si no existe
if VU is None:
    num_skus = D.shape[1]
    VU = np.ones(num_skus)
else:
    # asegurarse de que no haya NaNs y sea array float
    VU = np.array(VU, dtype=float)
    nan_count = np.isnan(VU).sum()
    if nan_count > 0:
        print(f"VU nan count before: {nan_count}, filling with 1.0")
        VU = np.nan_to_num(VU, nan=1.0)
        print("VU nan filled.")

# Inferir Sr si no existe
if Sr is None:
    NUM_SLOTS = 12
    NUM_RACKS = D_racks.shape[0]
    Sr = np.zeros((NUM_SLOTS, NUM_RACKS), dtype=int)
    for s in range(NUM_SLOTS):
        Sr[s, s % NUM_RACKS] = 1

slot_assignment = np.argmax(Sr, axis=1)
slot_assignments = [slot_assignment]

print('Running picking with inferred inputs...')
# --- Sanitizar D_racks ---
try:
    D_racks = np.array(D_racks, dtype=float)
    print('D_racks original shape:', D_racks.shape)
    # número de racks esperado según Sr
    expected_racks = int(np.max(np.argmax(Sr, axis=1))) + 1 if Sr.size > 0 else D_racks.shape[0]
    # recortar o ampliar para que sea cuadrada de tamaño expected_racks
    if D_racks.shape[0] != D_racks.shape[1]:
        m = min(D_racks.shape[0], D_racks.shape[1])
        D_racks = D_racks[:m, :m]
        print('Trimmed D_racks to square', D_racks.shape)
    if D_racks.shape[0] < expected_racks:
        # pad con valores grandes
        pad = expected_racks - D_racks.shape[0]
        fill_val = np.nanmax(D_racks[np.isfinite(D_racks)]) if np.isfinite(D_racks).any() else 1e5
        big = fill_val * 10
        D_new = np.full((expected_racks, expected_racks), big, dtype=float)
        D_new[:D_racks.shape[0], :D_racks.shape[1]] = D_racks
        D_racks = D_new
        print('Padded D_racks to', D_racks.shape)

    nan_count = int(np.isnan(D_racks).sum())
    if nan_count > 0:
        fill_val = np.nanmax(D_racks[np.isfinite(D_racks)]) if np.isfinite(D_racks).any() else 1e5
        fill_with = fill_val * 10
        print(f'D_racks NaN count: {nan_count}, filling NaNs with {fill_with}')
        D_racks = np.nan_to_num(D_racks, nan=fill_with)

    # Symmetrize
    D_racks = (D_racks + D_racks.T) / 2.0
    np.fill_diagonal(D_racks, 0.0)
    print('D_racks sanitized shape:', D_racks.shape)
except Exception as e:
    print('Warning: failed to sanitize D_racks:', e)

res = nsga2_picking_streamlit(
    slot_assignments=slot_assignments,
    D=D,
    VU=VU,
    Sr=Sr,
    D_racks=D_racks,
    pop_size=10,
    n_gen=5
)

for i, r in enumerate(res):
    print('\nResult for slot assignment', i)
    print('Dist total:', r['distancia_total'])
    print('Penalizado:', r['penalizado'])
    print('Rutas mejor individuo:')
    for j, ruta in enumerate(r['rutas_best']):
        print(' Pedido', j, ':', ' -> '.join(map(str, ruta)))

print('\nDone')
