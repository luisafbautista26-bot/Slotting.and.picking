import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import defaultdict
import traceback

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
    st.info("Por favor, sube el archivo Excel de parámetros (ejemplo: DATA_TES (1).xlsx)")
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
            # Si la hoja VU tiene una primera columna con el id del SKU, úsala.
            VU = {}
            for i, row in enumerate(VU_df.values):
                # row puede ser un array tipo [sku, volumen] o solo [volumen]
                try:
                    sku_key = int(row[0])
                    vol = float(row[1])
                except Exception:
                    # fallback: usar índice 1-based
                    sku_key = i + 1
                    vol = float(row[1]) if len(row) > 1 else float(row[0])
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
        - **f1 (Dispersión de racks):** Qué tan dispersos están los SKUs iguales en diferentes racks (menos dispersión es mejor).
        - **f2 (Costo de picking):** El costo total de recoger los productos, considerando la distancia y la demanda (menos es mejor).
        Cada punto azul en la gráfica es una solución eficiente (óptima en el sentido de Pareto).
        """)
        st.markdown("**Nota:** en la optimización de picking los puntos de descarga por defecto son los racks 0, 1, 2 y 3 (es decir, los 4 primeros índices) y el punto de inicio es el rack 0.")

        f1_vals = [f[0] for f in pareto_fitness]
        f2_vals = [f[1] for f in pareto_fitness]
        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(f1_vals, f2_vals, c="blue", s=50, label="Soluciones Pareto")
        ax.set_xlabel("f1 (Dispersión de racks)")
        ax.set_ylabel("f2 (Costo de picking)")
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
                        import numpy as _np
                        aug_list = _np.asarray(aug).tolist()
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
            ax.set_xlabel('Distancia Total')
            ax.set_ylabel('Distancia a SKUs más demandados')
            ax.set_title('Frente de Pareto Picking (todas las soluciones de slotting)')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Mostrar detalles de la mejor solución de cada slotting
            for idx, res in enumerate(resultados_picking):
                es_mejor_global = False
                if best_points:
                    best_overall_idx = np.argmin([f1+f2 for f1, f2, _ in best_points])
                    if idx == best_points[best_overall_idx][2]:
                        es_mejor_global = True
                st.markdown(f"#### Solución de slotting {idx+1} {'⭐' if es_mejor_global else ''}")
                st.write(f"**Mejor individuo:**\n- Distancia total: {res['distancia_total']:.2f}\n- Distancia a SKUs más demandados: {res['sku_distancia']:.2f}\n- Penalización: {'Sí' if res['penalizado'] else 'No'}" + ("\n\n:star: **Esta es la mejor solución global de picking**" if es_mejor_global else ""))
                st.write("**Rutas de picking por pedido (mejor individuo):**")
                rutas_best = res['rutas_best']

                # obtener racks de descarga usados por el solver para resaltarlos
                # preferir los racks devueltos por el solver en res (discharge_racks)
                try:
                    discharge_racks_ui = res.get('discharge_racks') or getattr(picking_solver, 'DISCHARGE_RACKS', [])
                except Exception:
                    discharge_racks_ui = []

                def format_route(route, discharge_racks_local):
                    # route es lista de ints; mostrar solo los números de racks (sin prefijo 'D')
                    # El usuario pidió que no se muestre la notación D(x) en la ruta.
                    out = [str(v) for v in route]
                    return ' → '.join(out)

                # Preferir mostrar el genoma aumentado completo (no separado por pedido).
                # Si no está disponible, mostrar la tabla de rutas por pedido como fallback.
                augmented_display = None
                try:
                    augmented_display = res.get('augmented_best')
                except Exception:
                    augmented_display = None

                # si no está en res, intentar extraerlo del individuo en population_eval_triples
                if augmented_display is None:
                    try:
                        pf = res.get('pareto_front', [])
                        if pf:
                            best_idx = pf[0][0]
                        else:
                            best_idx = 0
                        pet = res.get('population_eval_triples', {})
                        if pet and best_idx in pet:
                            augmented_display = pet[best_idx][4]
                    except Exception:
                        augmented_display = None

                # --- Replace the augmented_display handling with a more robust block ---
                if augmented_display is not None:
                    st.markdown('**Genome aumentado por pedido (cada fila muestra la secuencia con puntos de descarga):**')
                    try:
                        # canonicalize augmented_display into a Python list
                        seq = list(augmented_display)

                        # If augmented_display looks like a slot->sku assignment (length similar to NUM_SLOTS),
                        # rebuild a route-genome and re-evaluate to get the true augmented genome.
                        # This fixes cases where picking_solver accidentally returned the slotting assignment.
                        try:
                            # NUM_SLOTS is available in outer scope (from earlier when reading Sr)
                            is_slot_assign = isinstance(seq, list) and len(seq) >= NUM_SLOTS
                        except Exception:
                            is_slot_assign = False

                        if is_slot_assign:
                            # Rebuild genome and re-evaluate using picking_solver functions
                            try:
                                # slot_assign is the slotting solution we passed to picking; prefer to get the one used:
                                slot_assign_for_display = slot_assign if 'slot_assign' in locals() else seq
                                # Build a route-genome from the orders and this slot assignment
                                genome_rebuilt = picking_solver.build_genome(
                                    orders=np.array(D),
                                    slot_assignment_row=np.asarray(slot_assign_for_display),
                                    Vm_array_local=np.full(len(slot_assign_for_display), picking_solver.DEFAULT_VM_PER_SLOT),
                                    slot_to_rack_local=np.argmax(np.asarray(Sr), axis=1).tolist(),
                                    start_rack=0,
                                    cluster_idx=0,
                                    VU_map={i+1: VU[i] for i in range(len(VU))} if isinstance(VU, dict) is False else VU,
                                    D_racks=np.array(D_racks),
                                    PROHIBITED_SLOT_INDICES=list(PROHIBITED_SLOTS)
                                )
                                # Re-evaluate to get augmented genome
                                _, _, _, augmented_for_display = picking_solver.evaluate_individual(
                                    genome_rebuilt,
                                    [np.asarray(slot_assign_for_display)],
                                    np.argmax(np.asarray(Sr), axis=1).tolist(),
                                    box_volume_max=1.0,
                                    start_rack=0,
                                    orders=np.array(D),
                                    Vm_array_local=np.full(len(slot_assign_for_display), picking_solver.DEFAULT_VM_PER_SLOT),
                                    VU_map={i+1: VU[i] for i in range(len(VU))} if isinstance(VU, dict) is False else VU,
                                    D_racks=np.array(D_racks),
                                    PROHIBITED_SLOT_INDICES=list(PROHIBITED_SLOTS)
                                )
                                seq = list(augmented_for_display)
                            except Exception as e_rebuild:
                                # If rebuild fails, fall back to treating seq as-is (we'll still format best-effort)
                                print("DEBUG: rebuild genome failed:", e_rebuild)
                                seq = list(augmented_display)

                        # Now seq should be the augmented genome in the form [cluster_idx, 0, ..., 0, ...]
                        # Remove prefix [cluster_idx, 0] if present
                        body = seq[2:] if len(seq) >= 2 and seq[1] == 0 else seq[:]

                        # Mostrar también el genoma aumentado LITERALmente tal como lo devolvió el solver.
                        # El usuario pidió explícitamente que 0 inicie y termine y que no se muestre
                        # la etiqueta textual "(punto de descarga)", por lo que aquí imprimimos
                        # la secuencia tal cual (solo números) separada por ' → '.
                        try:
                            st.markdown('**Genoma aumentado (literal, tal como en `augmented_best`):**')
                            st.write(' → '.join(str(int(x)) for x in seq))
                        except Exception:
                            # si por algún motivo seq no es iterable o contiene tipos raros, hacer fallback seguro
                            try:
                                st.write(seq)
                            except Exception:
                                pass

                        # Derive discharge_racks to display. Prefer res value, then module default, then compute from PROHIBITED_SLOTS.
                        discharge_racks_ui = res.get('discharge_racks') if isinstance(res, dict) and res.get('discharge_racks') else getattr(picking_solver, 'DISCHARGE_RACKS', [])
                        # Ensure discharge racks do NOT include rack 0. User requires discharge racks to be 1,2,3 at minimum.
                        try:
                            discharge_racks_ui = [int(r) for r in discharge_racks_ui if int(r) != 0]
                        except Exception:
                            pass
                        if not discharge_racks_ui:
                            # compute from Sr & PROHIBITED_SLOTS as fallback (exclude rack 0)
                            try:
                                slot_to_rack_local = np.argmax(np.asarray(Sr), axis=1).tolist()
                                discharge_racks_ui = sorted(set(slot_to_rack_local[s] for s in PROHIBITED_SLOTS if 0 <= s < len(slot_to_rack_local) and slot_to_rack_local[s] != 0))
                            except Exception:
                                discharge_racks_ui = []
                        # ultimate fallback: ensure at least [1,2,3]
                        if not discharge_racks_ui:
                            discharge_racks_ui = [1, 2, 3]

                        # split by orders (0 separators) but treat consecutive zeros safely
                        # and limit the number of displayed pedidos to the number of
                        # non-empty orders actually present in D (esto evita que
                        # segmentos vacíos inflen el contador en la UI).
                        raw_segments = []
                        cur = []
                        for v in body:
                            if int(v) == 0:
                                raw_segments.append(cur)
                                cur = []
                            else:
                                cur.append(int(v))
                        if cur:
                            raw_segments.append(cur)

                        # número esperado de pedidos (filas no vacías en D)
                        try:
                            expected_orders = int(np.sum(np.any(np.asarray(D) != 0, axis=1)))
                        except Exception:
                            expected_orders = None

                        # Filtrar segmentos vacíos (producidos por ceros consecutivos)
                        orders_segments = [seg for seg in raw_segments if len(seg) > 0]

                        # Si hay más segmentos detectados que pedidos reales, truncar
                        if expected_orders is not None and len(orders_segments) > expected_orders:
                            orders_segments = orders_segments[:expected_orders]

                        # Mostrar un banner de diagnóstico compacto para depuración rápida
                        try:
                            zero_frac = sum(1 for x in seq if int(x) == 0) / len(seq) if seq else None
                        except Exception:
                            zero_frac = None
                        st.info(f"Diagnóstico: D.shape={np.asarray(D).shape}, pedidos_no_vacíos={expected_orders}, len(augmented)={len(seq)}, fracción_ceros={zero_frac}")

                        # Build display rows: show the augmented genome segments exactly
                        # as returned by the solver (incluye los racks de descarga
                        # que ya insertó `insert_discharge_points_and_boxes`). No
                        # intentamos insertar racks adicionales aquí; simplemente
                        # mostramos lo que devolvió el solver, para mantener la
                        # correspondencia exacta.
                        pedidos = []
                        rutas_aug = []
                        for i, seg in enumerate(orders_segments):
                            pedidos.append(f'Pedido {i+1}')
                            if len(seg) == 0:
                                rutas_aug.append('0 → 0')
                                continue
                            # mostrar la secuencia tal cual (los racks de descarga
                            # estarán presentes si el solver los insertó)
                            rutas_aug.append('0 → ' + ' → '.join(str(int(r)) for r in seg) + ' → 0')

                        rutas_tabla = pd.DataFrame({'Pedido': pedidos, 'Genome aumentado': rutas_aug})
                        st.dataframe(rutas_tabla)

                    except Exception as e_display:
                        print("DEBUG: error while formatting augmented_display:", e_display)
                        # fallback: show rutas_best (existing behavior)
                        rutas_tabla = pd.DataFrame({
                            'Pedido': [f'Pedido {i+1}' for i in range(len(rutas_best))],
                            'Ruta': [format_route(ruta, discharge_racks_ui) for ruta in rutas_best]
                        })
                        st.dataframe(rutas_tabla)
                else:
                    rutas_tabla = pd.DataFrame({
                        'Pedido': [f'Pedido {i+1}' for i in range(len(rutas_best))],
                        'Ruta': [format_route(ruta, discharge_racks_ui) for ruta in rutas_best]
                    })
                    st.dataframe(rutas_tabla)
        except Exception as e:
            st.error(f"Error al ejecutar picking: {e}")
            st.text(traceback.format_exc())

# --- Utilidades para depuración en la UI ---
st.markdown("---")
st.header("Herramientas de depuración")
col1, col2 = st.columns(2)
with col1:
    if st.button("Limpiar sesión (slotting_solutions)"):
        keys = ['slotting_solutions', 'D', 'VU', 'Sr', 'D_racks']
        for k in keys:
            if k in st.session_state:
                del st.session_state[k]
        st.success("Sesión limpiada: se han eliminado las soluciones de slotting de la sesión.")

with col2:
    if st.button("Recargar módulo picking_solver"):
        try:
            picking_solver = importlib.import_module('picking_solver')
            importlib.reload(picking_solver)
            st.success("Módulo 'picking_solver' recargado desde disco.")
        except Exception as e:
            st.error(f"No se pudo recargar 'picking_solver': {e}")

st.markdown("---")
st.subheader("Cargar y visualizar results_summary.json (desde disco)")
from pathlib import Path
res_path = Path('results_summary.json')
if st.button('Cargar results_summary.json'):
    if not res_path.exists():
        st.error('No se encontró results_summary.json en el workspace.')
    else:
        try:
            import json
            txt = res_path.read_text(encoding='utf-8')
            data = json.loads(txt)
            # construir resumen
            rows = []
            for e in data:
                idx = e.get('index')
                if 'result' in e and e['result']:
                    r = e['result']
                    dist = r.get('distancia_total')
                    sku_dist = r.get('sku_distancia')
                    penal = r.get('penalizado')
                    rutas = r.get('rutas_best') or []
                    n_rutas = len(rutas)
                else:
                    dist = None
                    sku_dist = None
                    penal = None
                    n_rutas = None
                rows.append({'index': idx, 'distancia_total': dist, 'sku_distancia': sku_dist, 'penalizado': penal, 'n_rutas': n_rutas})
            import pandas as _pd
            df = _pd.DataFrame(rows)
            st.dataframe(df.sort_values('index'))
            # ofrecer descarga del JSON tal cual
            st.download_button('Descargar results_summary.json', txt, file_name='results_summary.json', mime='application/json')
        except Exception as e:
            st.error(f'Error leyendo results_summary.json: {e}')
try:
    # Preferir mostrar el debug en la UI cuando la variable está disponible
    if 'resultados_picking' in locals() and resultados_picking:
        try:
            st.write("DEBUG - augmented_best:", resultados_picking[0].get('augmented_best'))
        except Exception:
            # Si por algún motivo st.write falla en este punto, caer al print de consola
            print("DEBUG - augmented_best (console):", resultados_picking[0].get('augmented_best'))
    else:
        # No hay resultados de picking en este pase de ejecución; imprimir nota en consola
        print("DEBUG - resultados_picking no está definido en el contexto actual (aún no se ejecutó picking)")
except Exception as e:
    # Protección extra: en caso de errores inesperados no romper la carga de la app
    try:
        print("DEBUG - error comprobando augmented_best:", e)
    except Exception:
        pass