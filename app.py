import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import milp, Bounds, LinearConstraint

st.set_page_config(page_title="Optimización Multiflujo - DataNet", layout="wide")

# ---------------------------------------------------------------- Datos base
EDGES = [(1, 2), (1, 3), (2, 3), (2, 4), (2, 5), (3, 4), (3, 6),
         (4, 5), (4, 7), (5, 6), (5, 8), (6, 7), (6, 8), (7, 8)]
COST = np.array([4, 6, 2, 5, 7, 3, 5, 2, 6, 3, 5, 2, 4, 3], dtype=float)
CAP_BASE = np.array([12, 10, 8, 10, 8, 12, 10, 10, 12, 8, 10, 10, 12, 14], dtype=float)
NAMES = [f"{i}→{j}" for i, j in EDGES]
IDX_36 = EDGES.index((3, 6))
NODES = range(1, 9)
FLOWS = [("F1", 1, 8), ("F2", 2, 7), ("F3", 3, 8)]
DEM_BASE = (6, 5, 4)
E, K = len(EDGES), len(FLOWS)

# Escenarios del enunciado: nombre -> (demandas, capacidades, descripción)
def make_scenarios():
    falla = CAP_BASE.copy(); falla[IDX_36] = 0
    ampl = CAP_BASE.copy(); ampl[IDX_36] = 14
    return {
        "Base": (DEM_BASE, CAP_BASE.copy(), "Datos originales."),
        "1. Falla": (DEM_BASE, falla, "Se desactiva el enlace 3→6 (capacidad 0)."),
        "2. Tráfico": ((10, 5, 4), CAP_BASE.copy(), "La demanda de F1 sube de 6 a 10 Gbps."),
        "3. Ampliación": (DEM_BASE, ampl, "La capacidad de 3→6 sube de 10 a 14 Gbps."),
    }


# ---------------------------------------------------------------- Modelo
def solve(demands, caps, integer=False):
    """Multiflujo de costo mínimo. Variable x[k*E + e] = flujo k en el arco e."""
    n = E * K
    c = np.tile(COST, K)

    A_eq = np.zeros((K * len(NODES), n))
    b_eq = np.zeros(K * len(NODES))
    for k, (_, o, d) in enumerate(FLOWS):
        for i in NODES:
            r = k * len(NODES) + (i - 1)
            for e, (a, b) in enumerate(EDGES):
                if a == i:
                    A_eq[r, k * E + e] += 1
                if b == i:
                    A_eq[r, k * E + e] -= 1
            b_eq[r] = demands[k] if i == o else (-demands[k] if i == d else 0)

    A_ub = np.zeros((E, n))
    for e in range(E):
        for k in range(K):
            A_ub[e, k * E + e] = 1

    cons = [LinearConstraint(A_eq, b_eq, b_eq),
            LinearConstraint(A_ub, -np.inf, caps)]
    res = milp(c=c, constraints=cons, bounds=Bounds(0, np.inf),
               integrality=np.ones(n) if integer else np.zeros(n))
    if not res.success:
        return None, res.message
    x = np.round(res.x, 6).reshape(K, E)
    return {"cost": float(res.fun), "x": x}, None


def result_table(sol, caps):
    x = sol["x"]
    total = x.sum(axis=0)
    df = pd.DataFrame({"Arco": NAMES, "Costo unit.": COST,
                       "F1": x[0], "F2": x[1], "F3": x[2],
                       "Flujo total": total, "Capacidad": caps})
    df["Uso (%)"] = np.where(caps > 0, total / np.where(caps > 0, caps, 1) * 100, 0.0)
    df["Saturado"] = np.isclose(total, caps) & (caps > 0)
    return df


# ---------------------------------------------------------------- Interfaz
st.title("Optimización multiflujo – DataNet Analytics")
st.write("Minimiza el costo total de transmisión de tres flujos sin superar las "
         "capacidades compartidas de los enlaces. Los flujos pueden dividirse por varias rutas.")

SC = make_scenarios()
st.sidebar.header("Configuración")
choice = st.sidebar.radio("Escenario", list(SC.keys()) + ["Personalizado"])
integer = st.sidebar.checkbox("Forzar flujos enteros", value=False,
                              help="El enunciado permite dividir flujos; sin marcar es el modelo continuo.")

if choice == "Personalizado":
    st.sidebar.subheader("Demandas (Gbps)")
    demands = tuple(st.sidebar.slider(f"{n}: {o} → {d}", 0, 20, DEM_BASE[k])
                    for k, (n, o, d) in enumerate(FLOWS))
    st.sidebar.subheader("Capacidades (Gbps)")
    caps = np.array([float(st.sidebar.number_input(NAMES[e], 0, 50, int(CAP_BASE[e]), key=f"cap{e}"))
                     for e in range(E)])
    desc = "Demandas y capacidades definidas manualmente."
else:
    demands, caps, desc = SC[choice]
    caps = caps.copy()

st.subheader(f"Escenario: {choice}")
st.caption(desc)
st.write("Demandas: " + ", ".join(f"**{n}** ({o}→{d}) = {demands[k]} Gbps"
                                  for k, (n, o, d) in enumerate(FLOWS)))

if st.button("Ejecutar optimización", use_container_width=True, type="primary"):
    sol, msg = solve(demands, caps, integer)
    if sol is None:
        st.error(f"No existe solución factible en este escenario: {msg}")
    else:
        df = result_table(sol, caps)
        st.success(f"Costo total mínimo: **{sol['cost']:.2f}** unidades monetarias")

        used = df[df["Flujo total"] > 1e-9].copy()
        sat = df[df["Saturado"]]["Arco"].tolist()
        c1, c2, c3 = st.columns(3)
        c1.metric("Costo total", f"{sol['cost']:.2f}")
        c2.metric("Demanda total", f"{sum(demands)} Gbps")
        c3.metric("Enlaces saturados", ", ".join(sat) if sat else "Ninguno")

        st.markdown("#### Flujo por enlace (solo enlaces usados)")
        show = used.drop(columns=["Saturado"])
        st.dataframe(show.style.format({"F1": "{:.2f}", "F2": "{:.2f}", "F3": "{:.2f}",
                                        "Flujo total": "{:.2f}", "Capacidad": "{:.0f}",
                                        "Uso (%)": "{:.1f}"})
                     .apply(lambda r: ["background-color: #ffd6d6" if r["Uso (%)"] >= 99.99 else ""
                                       for _ in r], axis=1),
                     use_container_width=True, hide_index=True)
        st.caption("Filas en rojo = enlace saturado. Los enlaces sin tráfico no se muestran.")
        st.markdown("#### Uso de capacidad por enlace")
        st.bar_chart(df.set_index("Arco")["Uso (%)"])

st.divider()
st.header("Comparación de los tres escenarios obligatorios")
rows = []
base_cost = None
for name, (dem, cp, d) in SC.items():
    sol, _ = solve(dem, cp, integer)
    cost = sol["cost"] if sol else np.nan
    if name == "Base":
        base_cost = cost
    sat = ""
    if sol:
        t = result_table(sol, cp)
        sat = ", ".join(t[t["Saturado"]]["Arco"])
    rows.append({"Escenario": name, "Modificación": d,
                 "Costo total": cost,
                 "Δ vs base": cost - base_cost if sol and base_cost == base_cost else np.nan,
                 "Δ %": (cost - base_cost) / base_cost * 100 if sol and base_cost else np.nan,
                 "Enlaces saturados": sat or "—",
                 "Factible": "Sí" if sol else "No"})
cmp_df = pd.DataFrame(rows)
st.dataframe(cmp_df.style.format({"Costo total": "{:.2f}", "Δ vs base": "{:+.2f}", "Δ %": "{:+.1f}%"}),
             use_container_width=True, hide_index=True)
st.bar_chart(cmp_df.set_index("Escenario")["Costo total"])
