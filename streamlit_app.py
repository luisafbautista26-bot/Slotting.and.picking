import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import defaultdict
import traceback
import re

# Helper global: parse an 'augmented' sequence into a list of integers robustly
def parse_sequence(maybe_seq):
    if maybe_seq is None:
        return None
    try:
        seq = list(maybe_seq)
    except Exception:
        seq = None
    if seq is not None:
        try:
            return [int(x) for x in seq]
        except Exception:
            pass
        if all(isinstance(x, str) and len(x) == 1 for x in seq):
            s = ''.join(seq)
            nums = re.findall(r'-?\d+', s)
            if nums:
                return [int(n) for n in nums]
    if isinstance(maybe_seq, str):
        nums = re.findall(r'-?\d+', maybe_seq)
        if nums:
            return [int(n) for n in nums]
    if seq is not None and all(isinstance(x, (tuple, list)) and len(x) >= 2 for x in seq):
        try:
            vals = [x[1] for x in seq]
            return [int(v) for v in vals]
        except Exception:
            pass
    return maybe_seq


def render_solution_block(title, distancia, sku_dist, penal, slot_assign_idx, augmented_obj, rutas_obj, raw_key_suffix=""):
    """Renderiza una solución usando el mismo formato que la sección 'Mejor solución'.
    - title: título o subtítulo a mostrar
    - distancia, sku_dist, penal: métricas (pueden ser None)
    - slot_assign_idx: índice de la solución de slotting (opcional)
    - augmented_obj: objeto que contiene el genoma aumentado (lista, np.array, string...)
    - rutas_obj: rutas por pedido (lista de secuencias o representaciones)
    - raw_key_suffix: sufijo único para checkbox de raw display
    """
    st.markdown(title)
    try:
        if distancia is not None:
            st.write(f"- Distancia total: {float(distancia):.2f}")
    except Exception:
        st.write(f"- Distancia total: {distancia}")
    try:
        if sku_dist is not None:
            st.write(f"- Distancia a SKUs más demandados: {float(sku_dist):.2f}")
    except Exception:
        st.write(f"- Distancia a SKUs más demandados: {sku_dist}")
    try:
        if penal is not None:
            st.write(f"- Penalización: {'Sí' if bool(penal) else 'No'}")
    except Exception:
        st.write(f"- Penalización: {penal}")
    try:
        if slot_assign_idx is not None:
            st.write(f"- Corresponde a la solución de slotting (cluster) #{int(slot_assign_idx)+1}")
    except Exception:
        pass

    st.markdown('**Genoma aumentado:**')
    seq = parse_sequence(augmented_obj)
    if seq is None:
        st.write('No hay genoma aumentado disponible.')
    else:
        try:
            st.write(' → '.join(str(int(x)) for x in seq))
        except Exception:
            st.write(seq)

    # Nota: se eliminó la opción de mostrar el genoma crudo aquí para
    # simplificar la interfaz. El genoma aumentado ya se muestra en formato
    # legible en la línea anterior.

    st.markdown('**Rutas por pedido:**')
    if not rutas_obj:
        st.write('No hay rutas disponibles.')
    else:
        for pid, ruta in enumerate(rutas_obj, start=1):
            parsed_route = None
            try:
                if isinstance(ruta, (np.ndarray, list, tuple)):
                    parsed_route = list(ruta)
                else:
                    parsed_route = parse_sequence(ruta)
            except Exception:
                try:
                    parsed_route = parse_sequence(ruta)
                except Exception:
                    parsed_route = None

            if parsed_route is None:
                st.write(f"Pedido {pid}: {ruta}")
            else:
                try:
                    st.write(f"Pedido {pid}: {' → '.join(str(int(x)) for x in parsed_route)}")
                except Exception:
                    st.write(f"Pedido {pid}: {parsed_route}")


def compute_slots_and_boxes_from_augmented(augmented, slot_assignment_row, slot_to_rack_local,
                                          orders, Vm_array_local, box_volume_max, discharge_racks, start_rack=0):
    """
    Simula el genoma aumentado y:
      - cuenta slots_with_product: slots del cluster con sku != 0
      - cuenta boxes_used:
          * cuenta descargas explícitas en `augmented` (racks en discharge_racks, EXCLUYENDO start_rack/0),
          * durante la simulación de picks unitarios, si un pick provocaría overflow se contabiliza (y reinicia) una caja ANTES del pick,
          * al encontrar un separador 0 (fin de pedido) se cuenta una caja si la caja contenía producto (box_vol > 0) y se reinicia box_vol.
      Garantía: box_vol nunca superará box_volume_max durante la simulación.
    """
    orders_arr = np.array(orders)
    order_demands = [dict((int(sku)+1, int(qty)) for sku, qty in enumerate(order) if qty > 0) for order in orders_arr]
    num_orders = len(order_demands)

    # slots with product in this cluster
    slots_with_product = sum(1 for s in slot_assignment_row if int(s) != 0)

    # exclude start_rack (0) from explicit discharge counting
    discharge_set = set(discharge_racks) - {start_rack}

    boxes_used = 0
    order_idx = 0
    i = 2
    box_vol = 0.0
    current_rack = start_rack

    # Necesitamos VU_map - si no existe globalmente, construirlo desde st.session_state
    VU_map_local = {}
    try:
        if 'VU_map' in globals():
            VU_map_local = globals()['VU_map']
        elif 'VU' in st.session_state:
            VU_data = st.session_state['VU']
            if isinstance(VU_data, dict):
                VU_map_local = {int(k): float(v) for k, v in VU_data.items()}
            else:
                VU_arr = np.asarray(VU_data, dtype=float)
                VU_map_local = {i+1: float(VU_arr[i]) for i in range(len(VU_arr))}
    except Exception:
        pass

    while i < len(augmented):
        rack = augmented[i]

        # end of order: count a box if box has content, then reset
        if rack == start_rack:
            if box_vol > 0:
                boxes_used += 1
            box_vol = 0.0
            order_idx += 1
            current_rack = start_rack
            i += 1
            continue

        # explicit discharge in augmented (excluding start_rack): count and reset
        if rack in discharge_set:
            boxes_used += 1
            box_vol = 0.0
            current_rack = rack
            i += 1
            continue

        # normal rack visit: simulate picks unit-by-unit
        if order_idx < num_orders:
            demand = order_demands[order_idx]
            for slot_idx, sku in enumerate(slot_assignment_row):
                if slot_idx >= len(slot_to_rack_local):
                    continue
                if slot_to_rack_local[slot_idx] != rack:
                    continue
                sku_id = int(sku)
                if sku_id == 0:
                    continue
                if sku_id not in demand or demand[sku_id] <= 0:
                    continue

                qty_to_pick = int(demand[sku_id])
                unit_vol = VU_map_local.get(sku_id, 0.0)
                if unit_vol <= 0:
                    demand[sku_id] = 0
                    continue

                # pick unit-by-unit: before each unit, if overflow would occur, count a box and reset
                for _ in range(qty_to_pick):
                    if box_vol + unit_vol > box_volume_max + 1e-12:
                        boxes_used += 1
                        box_vol = 0.0
                    box_vol += unit_vol
                demand[sku_id] = 0

        current_rack = rack
        i += 1

    # if remaining volume at end, count one more box
    if box_vol > 0:
        boxes_used += 1
        box_vol = 0.0

    return int(slots_with_product), int(boxes_used)


def extract_solution_info(info):
    """Normaliza diferentes formatos de 'info' y devuelve (distancia, sku_dist, penal, rutas_ind, augmented_ind)."""
    distancia = sku_dist = penal = rutas_ind = augmented_ind = None
    try:
        if isinstance(info, dict):
            distancia = info.get('distancia_total') or info.get('dist') or info.get('distance') or info.get('distancia')
            sku_dist = info.get('sku_distancia') or info.get('sku_dist') or info.get('sku_distance')
            penal = info.get('penalizado') if 'penalizado' in info else info.get('penal') or info.get('penalty')
            rutas_ind = info.get('rutas_best') or info.get('rutas') or info.get('routes') or info.get('rutas_por_pedido') or info.get('rutas_ind')
            augmented_ind = info.get('augmented') or info.get('augmented_best') or info.get('genome') or info.get('genoma')
        elif isinstance(info, (list, tuple)):
            # Different callers may pack evaluation tuples differently.
            # Known formats handled here:
            # 1) (dist, sku_dist, rutas, ind_obj, augmented)  <-- picking_solver current format
            # 2) (dist, sku_dist, penal, rutas)               <-- older/alternate format
            # 3) (dist, sku_dist, penal)                      <-- minimal tuple
            # 4) other: best-effort assignment
            try:
                if len(info) >= 5:
                    # Heuristic: if the 3rd element is a list (routes) and the 4th is a dict-like (individual),
                    # map to the picking_solver format.
                    third, fourth = info[2], info[3]
                    if isinstance(third, list) and (isinstance(fourth, dict) or hasattr(fourth, 'get')):
                        distancia, sku_dist, rutas_ind, _ind_obj, augmented_ind = info[:5]
                        penal = None
                    else:
                        # fallback to previous positional expectation
                        distancia, sku_dist, penal, rutas_ind, augmented_ind = info[:5]
                elif len(info) == 4:
                    distancia, sku_dist, penal, rutas_ind = info
                    augmented_ind = None
                elif len(info) == 3:
                    distancia, sku_dist, penal = info
                    rutas_ind = augmented_ind = None
                else:
                    rutas_ind = info
            except Exception:
                rutas_ind = info
        else:
            rutas_ind = info
    except Exception:
        rutas_ind = info
    return distancia, sku_dist, penal, rutas_ind, augmented_ind

st.title("NSGA-II Slotting & Picking Optimizer")

# Valores por defecto accesibles globalmente para evitar NameError si el usuario
# ejecuta picking antes de haber corrido el slotting.
PROHIBITED_SLOTS = {0, 1, 2, 3}

# === Instrucciones para el usuario ===
st.markdown("""
## Instrucciones para cargar el archivo Excel
El archivo Excel que subas debe contener las siguientes hojas, cada una con el formato adecuado:

**1. Hoja `VU`**  
Debe tener dos columnas:  
• La primera columna es el número de SKU (entero, por ejemplo: 1, 2, 3, ...).  
• La segunda columna es el volumen unitario de cada SKU (decimal, por ejemplo: 0.98, 0.32, ...).

**2. Hoja `D`**  
Debe ser una matriz donde cada fila representa un pedido y cada columna representa la cantidad de cada SKU en ese pedido.  
Por ejemplo, la columna 1 es el SKU 1, la columna 2 es el SKU 2, etc.

**3. Hoja `Sr`**  
Debe ser una matriz donde cada fila representa un slot y cada columna un rack.  
El valor es 1 si el slot pertenece a ese rack, 0 en caso contrario.

**4. Hoja `D_racks`**  
Debe ser una matriz cuadrada donde la posición (i, j) indica la distancia entre el rack i y el rack j.

Si alguna hoja no está presente o el formato no es correcto, la app mostrará un error.
""")

# --- Cargar archivo Excel ---
st.sidebar.header("Carga de parámetros")
excel_file = st.sidebar.file_uploader("Sube el archivo Excel de parámetros", type=["xlsx"])

if excel_file is None:
    st.info("Por favor, sube el archivo Excel de parámetros")
    st.stop()

# Leer todas las hojas
excel = pd.ExcelFile(excel_file)
st.sidebar.write(f"Hojas detectadas: {excel.sheet_names}")

# --- Selección de parámetros del algoritmo ---
st.sidebar.header("Parámetros del algoritmo NSGA-II")
pop_size = st.sidebar.number_input("Tamaño de población", min_value=10, max_value=200, value=30, step=1)
generations = st.sidebar.number_input("Generaciones", min_value=10, max_value=500, value=50, step=1)
cx_rate = st.sidebar.slider("Tasa de cruce (crossover)", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
pm_swap = st.sidebar.slider("Tasa de mutación (swap)", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
seed = st.sidebar.number_input("Semilla aleatoria", min_value=0, max_value=9999, value=42, step=1)

# --- Mostrar hojas y permitir selección de parámetros ---
st.header("Parámetros cargados desde Excel")
for sheet in excel.sheet_names:
    df = pd.read_excel(excel, sheet_name=sheet)
    st.subheader(f"Hoja: {sheet}")
    st.dataframe(df)

st.markdown("---")
st.header("Ejecución del algoritmo NSGA-II")

# --- Integración del algoritmo NSGA-II ---
import nsga2_solver
import importlib

if st.button("Ejecutar optimización"):
    try:
        if "VU" in excel.sheet_names:
            VU_df = pd.read_excel(excel, sheet_name="VU", header=None)
            VU_df = VU_df.dropna(how='all')
            VU_df = VU_df.dropna(axis=1, how='all')
        else:
            VU_df = None

        if "D" in excel.sheet_names:
            D_df = pd.read_excel(excel, sheet_name="D", header=None)
            D_df = D_df.dropna(how='all')
            D_df = D_df.dropna(axis=1, how='all')
        else:
            D_df = None

        if "D_racks" in excel.sheet_names:
            D_racks_df = pd.read_excel(excel, sheet_name="D_racks", header=None)
        else:
            D_racks_df = None
        Sr_df = pd.read_excel(excel, sheet_name="Sr") if "Sr" in excel.sheet_names else None
        D_racks_df = pd.read_excel(excel, sheet_name="D_racks") if "D_racks" in excel.sheet_names else None

        Vm = 3
        # No permitir al usuario elegir puntos prohibidos en slotting; se usan los valores fijos del código.
        # PROHIBITED_SLOTS para slotting se mantiene vacío (internamente picking puede recibir prohibiciones si se desea).
        user_prohibited = []

        if VU_df is not None:
            # Construir mapa VU con claves de SKU 1-based cuando sea posible.
            # La entrada en Excel puede venir con espacios no separables (\xa0)
            # o comas como separador decimal (por ejemplo '0,39'). Normalizamos.
            VU = {}
            def _parse_numeric_cell(x):
                """Intenta convertir una celda de Excel a float de forma robusta.
                - elimina espacios y \xa0
                - convierte coma decimal a punto
                - extrae el primer número válido si hay texto adicional
                """
                if x is None:
                    raise ValueError('None')
                # manejar ya-numeric
                try:
                    if isinstance(x, (int, float)):
                        return float(x)
                except Exception:
                    pass
                s = str(x).strip()
                # limpiar espacios no separables y espacios normales
                s = s.replace('\xa0', '').replace(' ', '')
                # normalizar coma decimal
                s = s.replace(',', '.')
                # extraer primera subcadena con formato numérico (opcional signo y decimales)
                import re
                m = re.search(r'-?\d+(?:\.\d+)?', s)
                if m:
                    return float(m.group(0))
                raise ValueError(f'No numeric value in cell: {x!r}')

            for i, row in enumerate(VU_df.values):
                # row puede venir como array de longitud variable; limpiamos NaNs
                row_list = [v for v in list(row) if not (isinstance(v, float) and np.isnan(v))]
                if not row_list:
                    continue
                # si hay al menos 2 columnas, asumimos [sku, vol], si no solo [vol]
                if len(row_list) >= 2:
                    sku_cell, vol_cell = row_list[0], row_list[1]
                else:
                    sku_cell, vol_cell = None, row_list[0]

                # parsear sku_key cuando sea posible
                if sku_cell is not None:
                    try:
                        sku_key_parsed = _parse_numeric_cell(sku_cell)
                        sku_key = int(sku_key_parsed)
                    except Exception:
                        sku_key = i + 1
                else:
                    sku_key = i + 1

                # parsear volumen
                try:
                    vol = _parse_numeric_cell(vol_cell)
                except Exception:
                    # último recurso: intentar float directo tras reemplazar comas
                    try:
                        vol = float(str(vol_cell).replace(',', '.').strip())
                    except Exception:
                        vol = 0.0

                VU[int(sku_key)] = float(vol)
        else:
            st.error("No se encontró la hoja 'VU' en el Excel.")
            st.stop()

        if D_df is not None:
            D = D_df.values
        else:
            st.error("No se encontró la hoja 'D' en el Excel.")
            st.stop()

        if VU_df is not None:
            num_vu = len(VU)
            num_d = D.shape[1]
            st.write(f"Índices de VU (mapeo SKU->volumen): {sorted(list(VU.keys()))}")
            st.write(f"Número de columnas en D (SKUs): {num_d}")
            expected_keys = list(range(1, num_d + 1))
            if sorted(list(VU.keys())) != expected_keys:
                st.error(f"Los índices de VU ({sorted(list(VU.keys()))}) no coinciden con los índices esperados (1 a {num_d}).\nRevisa que la hoja 'VU' tenga como primera columna el id de SKU o, si no, que las filas estén en el orden correcto; la app ahora espera claves 1-based para los SKUs.")
                st.stop()

        NUM_SKUS = D.shape[1]

        if Sr_df is not None:
            Sr = Sr_df.values
        else:
            st.error("No se encontró la hoja 'Sr' en el Excel.")
            st.stop()

        rack_assignment = np.argmax(Sr, axis=1)
        NUM_SLOTS = len(rack_assignment)
        rack_assignment = rack_assignment[:NUM_SLOTS]

        # Para picking, los puntos prohibidos son siempre los puntos de descarga
        # que nos indicaste: 0 es punto de inicio y 0,1,2,3 deben permanecer
        # prohibidos para asignación de SKUs.
        PROHIBITED_SLOTS = {0, 1, 2, 3}

        if D_racks_df is not None:
            D_racks = D_racks_df.values
        else:
            st.error("No se encontró la hoja 'D_racks' en el Excel.")
            st.stop()

        num_racks_d_racks = D_racks.shape[0]
        max_rack_assignment = np.max(rack_assignment)
        st.write(f"Tamaño de D_racks: {num_racks_d_racks}")
        st.write(f"Máximo índice de rack en rack_assignment: {max_rack_assignment}")
        if max_rack_assignment >= num_racks_d_racks:
            st.error(f"El máximo índice de rack usado ({max_rack_assignment}) es mayor o igual al tamaño de la matriz D_racks ({num_racks_d_racks}). Revisa que la matriz D_racks tenga suficientes filas/columnas para todos los racks usados en Sr.")
            st.stop()

        with st.spinner("Ejecutando NSGA-II, por favor espera..."):
            pareto_solutions, pareto_fitness = nsga2_solver.nsga2(
                pop_size=pop_size,
                generations=generations,
                cx_rate=cx_rate,
                pm_swap=pm_swap,
                seed=seed,
                NUM_SLOTS=NUM_SLOTS,
                PROHIBITED_SLOTS=list(PROHIBITED_SLOTS),
                NUM_SKUS=NUM_SKUS,
                D=D,
                VU=VU,
                Vm=Vm,
                rack_assignment=rack_assignment,
                D_racks=D_racks
            )

        st.success(f"¡Optimización completada! Se encontraron {len(pareto_solutions)} soluciones en el frente de Pareto.")

        st.markdown("""
        ### ¿Qué significa el resultado?
        El algoritmo NSGA-II busca **minimizar dos funciones objetivo**:
        - **Agrupamiento:** Se minimiza la distancia promedio entre los slots que almacenan el mismo SKU, esto garantiza que el mismo tipo de SKU se encuentre en la misma zona para facilitar la recolección de los productos.
        - **Priorización de la demanda:** Se determinan los SKUs más demandados y se minimiza la distancia del punto inicial a la zona donde se almacenan esos SKUs, garantizando que los productos más demandados se ubiquen en zonas más cercanas al punto inicial.
        
        Cada punto azul en la gráfica es una solución eficiente (óptima en el sentido de Pareto).
        """)

        f1_vals = [f[0] for f in pareto_fitness]
        f2_vals = [f[1] for f in pareto_fitness]
        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(f1_vals, f2_vals, c="blue", s=50, label="Soluciones Pareto")
        ax.set_xlabel("Agrupamiento")
        ax.set_ylabel("Priorización de la demanda")
        ax.set_title("Frente de Pareto - NSGA-II")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        fitness_array = np.array(pareto_fitness)
        ideal = fitness_array.min(axis=0)
        manhattan_distances = np.linalg.norm(fitness_array - ideal, ord=1, axis=1)
        best_manhattan_idx = np.argmin(manhattan_distances)
        st.subheader("Mejor solución balanceada (más cercana al ideal)")
        st.markdown("""
        **¿Cómo leer la asignación?**
        - El resultado es un vector donde **cada posición representa un slot** (espacio físico en el almacén).
        - El **valor en cada posición** es el número de SKU asignado a ese slot (por ejemplo, 0 significa slot vacío, 1 es SKU 1, 2 es SKU 2, etc.).
        - Así puedes ver fácilmente qué SKU va en cada slot.
        """)
        st.write(f"**Funciones objetivo:** f1 = {pareto_fitness[best_manhattan_idx][0]:.2f}, f2 = {pareto_fitness[best_manhattan_idx][1]:.2f}")
        st.write("**Asignación de SKUs a slots:**")
        asignacion = pareto_solutions[best_manhattan_idx]
        df_asignacion = pd.DataFrame({
            'Slot': np.arange(len(asignacion)),
            'SKU asignado': asignacion
        })
        st.dataframe(df_asignacion)

        # Guardar todas las soluciones del frente de Pareto de slotting para picking
        st.session_state['slotting_solutions'] = pareto_solutions
        st.session_state['D'] = D
        st.session_state['VU'] = VU
        st.session_state['Sr'] = Sr
        st.session_state['D_racks'] = D_racks

    except Exception as e:
        st.error(f"Error en la ejecución: {e}")
        st.text(traceback.format_exc())

    # Mostrar valores efectivos del módulo de picking (ayuda a verificar el estado en tiempo de ejecución)
    try:
        # intentar importar y recargar picking_solver para reflejar cambios en código sin reiniciar Streamlit
        picking_solver = importlib.import_module('picking_solver')
        importlib.reload(picking_solver)
        st.info(f"Valores en picking_solver: DISCHARGE_RACKS = {picking_solver.DISCHARGE_RACKS}, DEFAULT_VM_PER_SLOT = {picking_solver.DEFAULT_VM_PER_SLOT}, start_rack = 0")
    except Exception:
        # Fallback a un módulo mínimo que creamos para desbloquear la ejecución mientras se repara picking_solver.py
        try:
            picking_solver = importlib.import_module('picking_solver_minimal')
            importlib.reload(picking_solver)
            st.warning("Se está usando 'picking_solver_minimal' como respaldo; reinicia Streamlit cuando 'picking_solver' esté reparado para usar la implementación completa.")
            st.info(f"Valores en picking_solver (fallback): DISCHARGE_RACKS = {picking_solver.DISCHARGE_RACKS}, DEFAULT_VM_PER_SLOT = {picking_solver.DEFAULT_VM_PER_SLOT}, start_rack = 0")
        except Exception:
            st.warning("No se pudo importar ningún módulo de picking. Asegúrate de que 'picking_solver.py' o 'picking_solver_minimal.py' existan y no tengan errores de sintaxis.")

# --- Botón para ejecutar Picking (fuera del bloque de slotting) ---
if 'slotting_solutions' in st.session_state and len(st.session_state['slotting_solutions']) > 0:
    st.markdown("---")
    st.header("Optimización de Picking (NSGA-II)")
    st.markdown("""
    Ahora puedes optimizar el picking usando todas las soluciones de slotting encontradas.
    """)

    if st.button("Ejecutar optimización de picking para todas las soluciones de slotting"):
        st.info("Ejecutando NSGA-II de picking para todas las soluciones de slotting...")
        try:
            # Intentar importar la implementación completa; si falla, usar el módulo mínimo como respaldo
            try:
                picking_solver = importlib.import_module('picking_solver')
                importlib.reload(picking_solver)
            except Exception:
                picking_solver = importlib.import_module('picking_solver_minimal')
                importlib.reload(picking_solver)
                st.warning("Usando 'picking_solver_minimal' como respaldo para ejecutar picking.")

            # --- Validación previa: detectar pedidos que solo podrían ir a racks de descarga ---
            # Esto ayuda a evitar rutas tipo 0 -> descarga -> 0 cuando puede haber un
            # problema de asignación de SKUs a racks o slots prohibidos.
            slotting_solutions = st.session_state.get('slotting_solutions', [])
            # Obtener Sr y D desde la sesión (guardados al ejecutar slotting). Si no existen,
            # avisar al usuario y detener la ejecución del botón de picking.
            Sr_local = st.session_state.get('Sr')
            D_local = st.session_state.get('D')
            if Sr_local is None or D_local is None:
                st.error("Faltan datos requeridos para ejecutar picking: 'Sr' o 'D' no están en la sesión. Ejecuta primero el slotting y vuelve a intentar.")
                st.stop()
            slot_to_rack_local = np.argmax(Sr_local, axis=1).tolist()
            # derive discharge racks similarly a picking: from PROHIBITED_SLOTS -> racks
            discharge_racks_ui = sorted(set(slot_to_rack_local[s] for s in PROHIBITED_SLOTS if 0 <= s < len(slot_to_rack_local)))
            try:
                discharge_racks_ui = [int(r) for r in discharge_racks_ui if int(r) != 0]
            except Exception:
                pass
            if not discharge_racks_ui:
                discharge_racks_ui = [1, 2, 3]

            problematic_overall = []
            for sol_idx, slot_assign in enumerate(slotting_solutions):
                problems = []
                for order_idx, order in enumerate(D_local):
                    demand_keys = set(int(i+1) for i, q in enumerate(order) if q > 0)
                    # buscar al menos un slot (no prohibido) que tenga alguno de los SKUs
                    # y cuyo rack NO sea un rack de descarga
                    found_non_discharge = False
                    for slot_idx, sku in enumerate(slot_assign):
                        try:
                            sku_id = int(sku)
                        except Exception:
                            continue
                        if sku_id == 0:
                            continue
                        if sku_id not in demand_keys:
                            continue
                        if slot_idx in PROHIBITED_SLOTS:
                            continue
                        rack = slot_to_rack_local[slot_idx]
                        if rack not in discharge_racks_ui:
                            found_non_discharge = True
                            break
                    if not found_non_discharge:
                        problems.append(order_idx)
                if problems:
                    problematic_overall.append((sol_idx, problems))

            if problematic_overall:
                st.warning("Se detectaron pedidos que sólo podrían servirse desde racks de descarga en algunas soluciones de slotting. Revisa la lista antes de ejecutar picking.")
                for sol_idx, probs in problematic_overall:
                    st.write(f"- Solución de slotting {sol_idx+1}: pedidos potencialmente problemáticos (0-based indices): {probs}")
                ignore_and_run = st.checkbox("Ignorar advertencias y ejecutar picking de todos modos")
                if not ignore_and_run:
                    st.stop()

            resultados_picking = picking_solver.nsga2_picking_streamlit(
                slot_assignments=st.session_state['slotting_solutions'],
                D=st.session_state['D'],
                VU=st.session_state['VU'],
                Sr=st.session_state['Sr'],
                D_racks=st.session_state['D_racks'],
                pop_size=20,  # puedes ajustar
                n_gen=10,     # puedes ajustar
                prohibited_slots=list(PROHIBITED_SLOTS),
            )

            # ---- DEBUG helper (inspección rápida) ----
            try:
                res = resultados_picking if 'resultados_picking' in locals() else None
                if isinstance(res, list):
                    r0 = res[0]
                else:
                    r0 = res
                aug = None
                try:
                    aug = r0.get('augmented_best') if r0 is not None else None
                except Exception:
                    aug = None

                print("TYPE augmented_best:", type(aug))
                # Si es numpy array o lista con índices, convertir a lista simple
                try:
                    aug_list = list(aug)
                except Exception:
                    try:
                        aug_list = np.asarray(aug).tolist()
                    except Exception:
                        aug_list = aug

                print("LENGTH augmented:", None if aug_list is None else len(aug_list))
                print("FIRST 40 genes (compact):", aug_list[:40] if aug_list is not None else None)

                # intentar interpretar si es una asignación slot->sku (muchos índices con ceros intermitentes)
                zero_frac = sum(1 for x in aug_list if int(x) == 0) / len(aug_list) if aug_list else None
                print("Fracción de ceros en augmented:", zero_frac)

                # Si los valores parecen estar indexados como 'i:val' (tu salida), intentar reconstruir secuencia por valor:
                if aug_list and isinstance(aug_list[0], (tuple, list)) and len(aug_list[0]) == 2:
                    vals = [v for (_, v) in aug_list]
                    print("Reconstructed vals:", vals[:80])
            except Exception as _e:
                print('DEBUG helper failed:', _e)

            # nsga2_picking_streamlit returns a combined result dict when run across
            # all slotting solutions. Normalize to a list for downstream code that
            # expects an iterable of results per-slotting (for compatibility).
            if isinstance(resultados_picking, dict):
                resultados_picking = [resultados_picking]

            st.success(f"¡Optimización de picking completada para {len(resultados_picking)} conjuntos de soluciones de slotting (combinado)!")

            st.markdown("""
            ### ¿Qué significa el resultado?
            El algoritmo NSGA-II busca **minimizar dos funciones objetivo**:
            - **Distancia recorrida:** Se calcula la distancia entre cada nodo de la ruta, incluyendo a los puntos de descarga y al punto inicial, estas distancias se adicionan obteniendo la distancia total. En caso de que la ruta no satisfaga la demanda del pedido se aplica una penalización.
            - **Distancia SKUs frecuentes:** Esta función objetivo busca garantizar que los productos más demandados estén cerca al punto inicial. Para ello, se determina la demanda total de cada SKU y se seleccionan los 5 más demandados, luego se identifica qué slots del almacén contienen esos SKUs y se suma la distancia desde el punto inicial a los racks que los almacenan, esta distancia se minimiza.
            
            Cada punto azul en la gráfica es una solución eficiente (óptima en el sentido de Pareto).
            """)

            # Unificar la gráfica de Pareto de todas las soluciones: todos los puntos en azul, mejor solución con estrella dorada
            fig, ax = plt.subplots(figsize=(7,5))
            all_f1, all_f2 = [], []
            best_points = []
            for idx, res in enumerate(resultados_picking):
                pf = res['pareto_front']
                f1 = [x[1] for x in pf]
                f2 = [x[2] for x in pf]
                all_f1.extend(f1)
                all_f2.extend(f2)
                # Mejor de cada slotting
                best_idx = min(pf, key=lambda x: x[1])[0] if pf else 0
                best_f1 = res['population_eval_triples'][best_idx][0]
                best_f2 = res['population_eval_triples'][best_idx][1]
                best_points.append((best_f1, best_f2, idx))
            # Todos los puntos en azul
            ax.scatter(all_f1, all_f2, color='tab:blue', s=60, alpha=0.7, label='Soluciones Pareto')
            # Mejor solución global con estrella dorada
            if best_points:
                best_overall_idx = np.argmin([f1+f2 for f1, f2, _ in best_points])
                best_f1, best_f2, best_idx = best_points[best_overall_idx]
                ax.scatter([best_f1], [best_f2], color='gold', s=180, marker='*', edgecolor='black', label='Mejor global')
            ax.set_xlabel('Distancia recorrida')
            ax.set_ylabel('Distancia SKUs frecuentes')
            ax.set_title('Frente de Pareto Picking (todas las soluciones de slotting)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Mostrar la mejor solución global primero y luego una lista reducida
            # de las mejores 10 soluciones para facilitar la inspección del usuario.
            best_overall_slot_idx = None
            if best_points:
                best_overall_idx = int(np.argmin([f1+f2 for f1, f2, _ in best_points]))
                _, _, best_overall_slot_idx = best_points[best_overall_idx]

            # Nota: no renderizamos la mejor solución aquí porque la mejor
            # debe calcularse a partir de la lista combinada de individuos
            # (esto nos permite excluirla del listado desplegable que viene
            # a continuación). La mejor solución se mostrará más abajo.

            # Construir lista combinada de todas las soluciones del frente de Pareto
            combined = []
            for slot_idx, res in enumerate(resultados_picking):
                pf = res.get('pareto_front', [])
                pet = res.get('population_eval_triples', {})
                for entry in pf:
                    try:
                        ind_idx, f1, f2 = entry[0], entry[1], entry[2]
                    except Exception:
                        continue
                    combined.append({'slot_idx': slot_idx, 'ind_idx': ind_idx, 'f1': f1, 'f2': f2, 'sum': f1 + f2, 'res': res})

            # Todas las soluciones que queremos mostrar son las que aparecen en
            # los frentes de Pareto de cada ejecución (las entradas de 'combined').
            # Ordenarlas por (f1, f2) para presentación. Mostraremos la mejor
            # solución por separado y luego todas las demás en expanders.
            combined_sorted = sorted(combined, key=lambda x: (x['f1'], x['f2']))

            # Calcular la mejor solución global a partir de la lista combinada
            best_entry = None
            if combined:
                try:
                    best_entry = min(combined, key=lambda x: x['sum'])
                except Exception:
                    best_entry = None

            # Mostrar mejor solución global (formato idéntico al de los demás)
            if best_entry is not None:
                try:
                    slot_idx = best_entry['slot_idx']
                    ind_idx = best_entry['ind_idx']
                    res = best_entry['res']
                    pet = res.get('population_eval_triples', {})
                    info = pet.get(ind_idx)
                    if info is not None:
                        distancia, sku_dist, penal, rutas_ind, augmented_ind = extract_solution_info(info)
                    else:
                        distancia = sku_dist = penal = rutas_ind = augmented_ind = None
                    
                    # Calcular slots y cajas usados
                    slots_used_count = None
                    boxes_used_count = None
                    if augmented_ind is not None:
                        try:
                            slotting_solutions = st.session_state.get('slotting_solutions', [])
                            Sr_local = st.session_state.get('Sr')
                            D_local = st.session_state.get('D')
                            if slotting_solutions and Sr_local is not None and D_local is not None and 0 <= slot_idx < len(slotting_solutions):
                                slot_assignment_row = slotting_solutions[slot_idx]
                                slot_to_rack_local = np.argmax(Sr_local, axis=1).tolist()
                                discharge_racks_calc = res.get('discharge_racks', [1, 2, 3])
                                Vm_array_local = np.full(len(slot_assignment_row), 3)  # DEFAULT_VM_PER_SLOT
                                box_volume_max = 1.0
                                slots_used_count, boxes_used_count = compute_slots_and_boxes_from_augmented(
                                    augmented_ind,
                                    slot_assignment_row,
                                    slot_to_rack_local,
                                    D_local,
                                    Vm_array_local,
                                    box_volume_max,
                                    discharge_racks_calc,
                                    start_rack=0
                                )
                        except Exception as e:
                            st.warning(f"No se pudo calcular slots/cajas: {e}")
                    
                    render_solution_block('## Mejor solución global (resumen) ⭐', distancia, sku_dist, penal, slot_idx, augmented_ind, rutas_ind, raw_key_suffix=f"best_{slot_idx}_{ind_idx}")
                    
                    # Mostrar slots y cajas si están disponibles
                    if slots_used_count is not None and boxes_used_count is not None:
                        st.write(f"**Cantidad de slots usados:** {slots_used_count}")
                        st.write(f"**Número de cajas utilizadas:** {boxes_used_count}")
                    
                    # Mostrar la asignación de slots (solución de slotting) usada
                    try:
                        slotting_solutions = st.session_state.get('slotting_solutions', [])
                        if slotting_solutions and 0 <= slot_idx < len(slotting_solutions):
                            slot_assign_used = slotting_solutions[slot_idx]
                            st.subheader('Asignación de slots usada (solución de slotting)')
                            df_assign = pd.DataFrame({
                                'Slot': np.arange(len(slot_assign_used)),
                                'SKU asignado': slot_assign_used
                            })
                            st.dataframe(df_assign)
                    except Exception:
                        pass
                except Exception:
                    pass

            st.markdown('## Otras soluciones (desplegables)')
            # Mostrar todas las soluciones que aparecen en los frentes de Pareto
            # (excluyendo la mejor ya mostrada) en expanders.
            for rank, entry in enumerate(combined_sorted, start=1):
                slot_idx = entry['slot_idx']
                ind_idx = entry['ind_idx']
                f1 = entry['f1']
                f2 = entry['f2']
                res = entry['res']

                # Omitir la entrada que ya mostramos como mejor solución
                if best_entry is not None and slot_idx == best_entry.get('slot_idx') and ind_idx == best_entry.get('ind_idx'):
                    continue

                with st.expander(f"#{rank} - Slotting {slot_idx+1} · f1={f1:.2f}, f2={f2:.2f}"):
                    pet = res.get('population_eval_triples', {})
                    info = pet.get(ind_idx)
                    if info is None:
                        st.write('No hay detalles disponibles para este individuo.')
                    else:
                        distancia, sku_dist, penal, rutas_ind, augmented_ind = extract_solution_info(info)
                        render_solution_block('', distancia, sku_dist, penal, slot_idx, augmented_ind, rutas_ind, raw_key_suffix=f"nd_{slot_idx}_{ind_idx}")

                            # Adjacent discharge->discharge cases are handled inside the
                            # solver by collapsing consecutive discharge visits. The
                            # UI no longer emits a warning here.
                
        except Exception as e:
            st.error(f"Error al ejecutar picking: {e}")
            st.text(traceback.format_exc())

# Se ha eliminado la sección de carga/visualización de results_summary.json
# para limpiar la interfaz. Si necesitas volver a habilitarla, podemos
# reintroducirla bajo una opción de depuración.
# Removed verbose debug display of 'augmented_best' to keep the UI clean.
# If you need to enable this debug output again, re-add a controlled
# debug checkbox and display it conditionally to avoid cluttering the UI.