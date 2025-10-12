import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import defaultdict
import traceback

st.title("NSGA-II Slotting & Picking Optimizer")

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
        PROHIBITED_SLOTS = {0, 1, 2, 3}

        if VU_df is not None:
            VU = {i: float(row[1]) for i, row in enumerate(VU_df.values)}
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
            st.write(f"Índices de VU: {list(VU.keys())}")
            st.write(f"Número de columnas en D: {num_d}")
            if list(VU.keys()) != list(range(num_d)):
                st.error(f"Los índices de VU ({list(VU.keys())}) no coinciden con los índices esperados (0 a {num_d-1}).\nRevisa que no haya filas vacías o desfasadas en la hoja VU y que el número de columnas en D sea correcto.")
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
                PROHIBITED_SLOTS=PROHIBITED_SLOTS,
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
            import picking_solver
            resultados_picking = picking_solver.nsga2_picking_streamlit(
                slot_assignments=st.session_state['slotting_solutions'],
                D=st.session_state['D'],
                VU=st.session_state['VU'],
                Sr=st.session_state['Sr'],
                D_racks=st.session_state['D_racks'],
                pop_size=20,  # puedes ajustar
                n_gen=10      # puedes ajustar
            )
            st.success(f"¡Optimización de picking completada para {len(st.session_state['slotting_solutions'])} soluciones de slotting!")

            # Unificar la gráfica de Pareto de todas las soluciones
            fig, ax = plt.subplots(figsize=(7,5))
            colores = plt.cm.tab10.colors
            best_points = []
            for idx, res in enumerate(resultados_picking):
                pf = res['pareto_front']
                f1 = [x[1] for x in pf]
                f2 = [x[2] for x in pf]
                ax.scatter(f1, f2, label=f'Slotting {idx+1}', color=colores[idx % len(colores)], s=60, alpha=0.6)
                # Resaltar la mejor solución de cada slotting
                best_idx = min(pf, key=lambda x: x[1])[0] if pf else 0
                best_f1 = res['population_eval_triples'][best_idx][0]
                best_f2 = res['population_eval_triples'][best_idx][1]
                best_points.append((best_f1, best_f2, idx))
            # Dibujar la mejor solución de todas (la más cercana al ideal)
            if best_points:
                best_overall_idx = np.argmin([f1+f2 for f1, f2, _ in best_points])
                best_f1, best_f2, best_idx = best_points[best_overall_idx]
                ax.scatter([best_f1], [best_f2], color='red', s=120, marker='*', label='Mejor global')
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
                rutas_tabla = pd.DataFrame({
                    'Pedido': [f'Pedido {i+1}' for i in range(len(rutas_best))],
                    'Ruta': [' → '.join(map(str, ruta)) for ruta in rutas_best]
                })
                st.dataframe(rutas_tabla)
        except Exception as e:
            st.error(f"Error al ejecutar picking: {e}")
            st.text(traceback.format_exc())
