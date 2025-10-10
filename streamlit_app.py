import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import defaultdict
from sklearn.cluster import KMeans

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
    # --- Ejemplo de mapeo: se asume que las hojas tienen nombres estándar ---
    # Puedes adaptar esto según la estructura real de tu Excel
    try:
        # Ejemplo: leer parámetros desde hojas específicas
        # Aquí debes adaptar el mapeo según tu Excel
        VU_df = pd.read_excel(excel, sheet_name="VU") if "VU" in excel.sheet_names else None
        D_df = pd.read_excel(excel, sheet_name="D") if "D" in excel.sheet_names else None
        Sr_df = pd.read_excel(excel, sheet_name="Sr") if "Sr" in excel.sheet_names else None
        D_racks_df = pd.read_excel(excel, sheet_name="D_racks") if "D_racks" in excel.sheet_names else None

        # --- Parámetros fijos o desde Excel ---
        Vm = 3
        PROHIBITED_SLOTS = {0, 1, 2, 3}

        # --- Procesar VU ---
        if VU_df is not None:
            VU = {int(row[0]): float(row[1]) for row in VU_df.values}
        else:
            st.error("No se encontró la hoja 'VU' en el Excel.")
            st.stop()

        # --- Procesar D (demanda) ---
        if D_df is not None:
            D = D_df.values
        else:
            st.error("No se encontró la hoja 'D' en el Excel.")
            st.stop()

        NUM_SKUS = D.shape[1]

        # --- Procesar Sr (racks por slot) ---
        if Sr_df is not None:
            Sr = Sr_df.values
        else:
            st.error("No se encontró la hoja 'Sr' en el Excel.")
            st.stop()

        rack_assignment = np.argmax(Sr, axis=1)
        NUM_SLOTS = len(rack_assignment)
        rack_assignment = rack_assignment[:NUM_SLOTS]

        # --- Procesar D_racks (distancias entre racks) ---
        if D_racks_df is not None:
            D_racks = D_racks_df.values
        else:
            st.error("No se encontró la hoja 'D_racks' en el Excel.")
            st.stop()

        # --- Ejecutar NSGA-II ---
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

        # --- Visualización del frente de Pareto ---
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

        # --- Mejor solución balanceada (distancia Manhattan al ideal) ---
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

        # --- Soluciones representativas por clustering ---
        st.subheader("Soluciones representativas (clustering)")
        st.markdown("""
        Se muestran ejemplos de soluciones distintas encontradas por el algoritmo, agrupadas por similitud.
        """)
        slotting_solutions = []
        try:
            if len(pareto_fitness) >= 3:
                kmeans = KMeans(n_clusters=3, random_state=0).fit(fitness_array)
                labels = kmeans.labels_
                for cluster in range(3):
                    idxs = np.where(labels == cluster)[0]
                    if len(idxs) > 0:
                        idx = idxs[0]
                        st.write(f"Cluster {cluster+1}:")
                        st.write(f"Funciones objetivo: f1 = {pareto_fitness[idx][0]:.2f}, f2 = {pareto_fitness[idx][1]:.2f}")
                        st.write("Asignación de SKUs a slots:")
                        df_asig = pd.DataFrame({
                            'Slot': np.arange(len(pareto_solutions[idx])),
                            'SKU asignado': pareto_solutions[idx]
                        })
                        st.dataframe(df_asig)
                        slotting_solutions.append(pareto_solutions[idx])
            else:
                st.info("No hay suficientes soluciones para clustering.")
                slotting_solutions = pareto_solutions
            # Guardar soluciones de slotting en el estado de sesión (dentro del try)
            st.session_state['slotting_solutions'] = slotting_solutions
            st.session_state['D'] = D
            st.session_state['VU'] = VU
            st.session_state['Sr'] = Sr
            st.session_state['D_racks'] = D_racks
        except Exception as e:
            st.error(f"Error en la ejecución: {e}")

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
            resultados_picking = picking_solver.nsga2_picking(
                slot_assignments=st.session_state['slotting_solutions'],
                D=st.session_state['D'],
                VU=st.session_state['VU'],
                Sr=st.session_state['Sr'],
                D_racks=st.session_state['D_racks'],
                pop_size=10,  # puedes ajustar
                n_gen=5       # puedes ajustar
            )
            st.success(f"¡Optimización de picking completada para {len(st.session_state['slotting_solutions'])} soluciones de slotting!")
            for idx, res in enumerate(resultados_picking):
                st.markdown(f"#### Solución de slotting {idx+1}")
                st.write(f"Funciones objetivo de picking: distancia total = {res['distancia_total']:.2f}, distancia SKUs demandados = {res['sku_distancia']:.2f}")
                st.write("Ruta de picking por pedido:")
                for pid, ruta in enumerate(res['rutas'], start=1):
                    st.write(f"Pedido {pid}: {' → '.join(map(str, ruta))}")
        except Exception as e:
            st.error(f"Error al ejecutar picking: {e}")
except Exception as e:
st.error(f"An error occurred: {e}")
