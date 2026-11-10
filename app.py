import streamlit as st
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
import pandas as pd

st.set_page_config(page_title="Optimización de Flujo de Redes", layout="wide")

st.title("🕸️ Optimización Multi-Flujo con MILP")
st.write("Esta aplicación optimiza de forma independiente y conjunta el flujo de tres demandas (F1, F2, F3) a través de una red compartida, minimizando el costo total de transporte sin superar las capacidades de los arcos.")

# --- Configuración de parámetros en la barra lateral ---
st.sidebar.header("Parámetros de Demanda")
flow_1_qty = st.sidebar.slider("Demanda Flujo 1 (Nodo 1 -> Nodo 8)", 0, 15, 6, 1)
flow_2_qty = st.sidebar.slider("Demanda Flujo 2 (Nodo 2 -> Nodo 7)", 0, 15, 5, 1)
flow_3_qty = st.sidebar.slider("Demanda Flujo 3 (Nodo 3 -> Nodo 8)", 0, 15, 4, 1)

# --- Definición del Problema ---
edge_names = [
    "X12", "X13", "X23", "X24", "X25", "X34", "X36", "X45", "X47", "X56", "X58", "X67", "X68", "X78"
]
num_edges = len(edge_names)

c_base = np.array([4, 6, 2, 5, 7, 3, 5, 2, 6, 3, 5, 2, 4, 3])
c = np.concatenate([c_base, c_base, c_base])

bounds = Bounds(0, 14)
integrality = np.ones_like(c)

A_flow = np.array([
    [ 1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # Nodo 1
    [-1,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0], # Nodo 2
    [ 0, -1, -1,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0], # Nodo 3
    [ 0,  0,  0, -1,  0, -1,  0,  1,  1,  0,  0,  0,  0,  0], # Nodo 4
    [ 0,  0,  0,  0, -1,  0,  0, -1,  0,  1,  1,  0,  0,  0], # Nodo 5
    [ 0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0,  1,  1,  0], # Nodo 6
    [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0,  1], # Nodo 7
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1, -1]  # Nodo 8
])

A_eq = np.zeros((24, num_edges * 3))
A_eq[0:8, 0:num_edges] = A_flow
A_eq[8:16, num_edges:2*num_edges] = A_flow
A_eq[16:24, 2*num_edges:3*num_edges] = A_flow

b_eq_F1 = np.array([flow_1_qty, 0, 0, 0, 0, 0, 0, -flow_1_qty])
b_eq_F2 = np.array([0, flow_2_qty, 0, 0, 0, 0, -flow_2_qty, 0])
b_eq_F3 = np.array([0, 0, flow_3_qty, 0, 0, 0, 0, -flow_3_qty])
b_eq = np.concatenate([b_eq_F1, b_eq_F2, b_eq_F3])

capacities = np.array([12, 10, 8, 10, 8, 12, 10, 10, 12, 8, 10, 10, 12, 14])
A_ub = np.zeros((num_edges, num_edges * 3))
for i in range(num_edges):
    A_ub[i, i] = 1
    A_ub[i, i + num_edges] = 1
    A_ub[i, i + 2*num_edges] = 1

A_all = np.vstack([A_eq, A_ub])
b_l = np.concatenate([b_eq, -np.inf * np.ones(num_edges)])
b_u = np.concatenate([b_eq, capacities])

constraints = LinearConstraint(A_all, b_l, b_u)

# --- Resolver ---
if st.button("⚡ Ejecutar Optimización", use_container_width=True):
    res = milp(c=c, bounds=bounds, constraints=constraints, integrality=integrality)
    
    if res.success:
        st.success(f"¡Optimización Exitosa! Costo Total Mínimo: **{res.fun:.2f}**")
        
        # Construir el DataFrame de Resultados
        f1_flows = [round(res.x[i]) for i in range(num_edges)]
        f2_flows = [round(res.x[i + num_edges]) for i in range(num_edges)]
        f3_flows = [round(res.x[i + 2*num_edges]) for i in range(num_edges)]
        total_flows = [f1 + f2 + f3 for f1, f2, f3 in zip(f1_flows, f2_flows, f3_flows)]
        
        df = pd.DataFrame({
            'Arco': edge_names,
            'Costo Unitario': c_base,
            'Flujo F1': f1_flows,
            'Flujo F2': f2_flows,
            'Flujo F3': f3_flows,
            'Flujo Total': total_flows,
            'Capacidad': capacities
        })
        
        st.dataframe(df, use_container_width=True)
        
        # Mostrar algunas métricas resumidas
        col1, col2, col3 = st.columns(3)
        col1.metric("Suma Total de Flujos", sum(total_flows))
        col2.metric("Costo Medio por Unidad", f"{res.fun / max(1, sum(total_flows)):.2f}")
        col3.metric("Uso de Capacidad Promedio", f"{(sum(total_flows)/sum(capacities))*100:.1f}%")
    else:
        st.error(f"No se encontró solución factible: {res.message}")
