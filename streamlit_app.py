

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import defaultdict
from sklearn.cluster import KMeans

st.title("NSGA-II Slotting & Picking Optimizer")

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
        st.write(f"Fitness: {pareto_fitness[best_manhattan_idx]}")
        st.write(f"Asignación: {pareto_solutions[best_manhattan_idx]}")

        # --- Soluciones representativas por clustering ---
        st.subheader("Soluciones representativas (clustering)")
        if len(pareto_fitness) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=0).fit(fitness_array)
            labels = kmeans.labels_
            for cluster in range(3):
                idxs = np.where(labels == cluster)[0]
                if len(idxs) > 0:
                    idx = idxs[0]
                    st.write(f"Cluster {cluster+1}:")
                    st.write(f"Fitness: {pareto_fitness[idx]}")
                    st.write(f"Asignación: {pareto_solutions[idx]}")
        else:
            st.info("No hay suficientes soluciones para clustering.")
    except Exception as e:
        st.error(f"Error en la ejecución: {e}")
