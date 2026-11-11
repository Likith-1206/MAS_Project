"""Streamlit dashboard:  streamlit run src/dashboard.py
Tabs: (1) network replay - queues and signal phases per intersection over time, (2) comparison plots/tables, (3) about."""
import glob, json, os
import numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, T = os.path.join(ROOT, "results"), os.path.join(ROOT, "traces")
NAMES = {"fixed_time": "Fixed-time", "max_pressure": "Max-Pressure", "independent": "Independent DQN", "coop": "Cooperative DQN"}
st.set_page_config(page_title="Traffic signal MARL", layout="wide")
st.title("Multi-agent traffic signal coordination - 3x3 grid")
tab1, tab2, tab3 = st.tabs(["Network replay", "Results", "About"])


def grid_xy(agent):  # SUMO ids: A0..C2 ; letter = row (A top), digit = column
    return "012".index(agent[1]), "CBA".index(agent[0])


with tab1:
    files = sorted(glob.glob(os.path.join(T, "*.json")))
    if not files:
        st.warning("No traces found. Run `bash run_all.sh` first.")
    else:
        pick = st.selectbox("Controller trace (rush-hour scenario, eval seed 100)", files,
                            format_func=lambda f: NAMES.get(os.path.basename(f).split("_s")[0], os.path.basename(f)))
        tr = json.load(open(pick)); steps = tr["steps"]
        i = st.slider("Time step (each = 5 s)", 0, len(steps) - 1, len(steps) // 2)
        s = steps[i]; ag = tr["agents"]
        xs, ys = zip(*[grid_xy(a) for a in ag])
        q = [s["queue"][a] for a in ag]
        fig = go.Figure()
        for a in ag:   # links between neighbours
            x, y = grid_xy(a)
            for b in ag:
                x2, y2 = grid_xy(b)
                if abs(x - x2) + abs(y - y2) == 1 and a < b:
                    fig.add_trace(go.Scatter(x=[x, x2], y=[y, y2], mode="lines", line=dict(color="#bbb", width=6), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=[f"{a}<br>q={v:.0f}<br>{'NS' if s['phase'][a] == 0 else 'EW'}" for a, v in zip(ag, q)],
                                 marker=dict(size=[30 + 3 * v for v in q], color=q, colorscale="RdYlGn_r", cmin=0, cmax=40, showscale=True,
                                             colorbar=dict(title="queue")), textposition="middle center", showlegend=False))
        fig.update_layout(height=520, xaxis=dict(visible=False, range=[-.6, 2.6]), yaxis=dict(visible=False, range=[-.6, 2.6], scaleanchor="x"),
                          title=f"t = {s['t']} s   total queue = {sum(q):.0f} vehicles   (label: green direction NS or EW)")
        st.plotly_chart(fig, width="stretch")
        tot = [sum(x["queue"].values()) for x in steps]
        line = go.Figure(go.Scatter(x=[x["t"] for x in steps], y=tot, mode="lines")); line.add_vline(x=s["t"])
        line.update_layout(height=240, xaxis_title="time (s)", yaxis_title="total queue (veh)")
        st.plotly_chart(line, width="stretch")

with tab2:
    for f, cap in [("comparison_main.png", "Main comparison"), ("ablation_communication.png", "RQ2 ablation"), ("learning_curves.png", "Learning curves")]:
        p = os.path.join(R, f)
        if os.path.exists(p): st.image(p, caption=cap)
    for f in ["summary_table.csv", "stat_tests.csv", "summary_per_scenario.csv"]:
        p = os.path.join(R, f)
        if os.path.exists(p): st.subheader(f); st.dataframe(pd.read_csv(p), width="stretch")

with tab3:
    st.markdown("Each intersection is an agent; observation = own lane queues/counts, phase, time-in-phase (+ optional neighbour info); "
                "action = next green phase; reward = negative queue (+ neighbour-mixed reward for the cooperative agent). "
                "See `docs/` and `README.md`.")
