"""Analyst UI: users ranked by GNN risk score, with the selected user's neighborhood from Neo4j."""
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from itgnn.neo4j_io import get_driver, neighborhood

RESULTS = Path("results")

st.set_page_config(page_title="Insider threat analyst view", layout="wide")
st.title("Insider threat analyst view")

scores = pd.read_csv(RESULTS / "gnn_scores.csv", index_col="user")
baseline = RESULTS / "baseline_scores.csv"
if baseline.exists():
    scores = scores.join(pd.read_csv(baseline, index_col="user")[["baseline_score"]])
scores = scores.sort_values("gnn_score", ascending=False)

left, right = st.columns([1, 2])

with left:
    st.subheader("Ranked users")
    user = st.radio(
        "Select a user",
        scores.index[:25],
        format_func=lambda u: f"{u}  ({scores.loc[u, 'gnn_score']:.2f})",
    )

with right:
    row = scores.loc[user]
    c1, c2, c3 = st.columns(3)
    c1.metric("GNN risk score", f"{row['gnn_score']:.2f}")
    if "baseline_score" in row:
        c2.metric("Baseline score", f"{row['baseline_score']:.2f}")
    c3.metric("Ground truth", "Malicious" if row["label"] else "Normal")

    try:
        driver = get_driver()
        rows = neighborhood(driver, user)
    except Exception as exc:  # Neo4j not running or not loaded yet
        st.warning(f"Could not read the graph from Neo4j: {exc}")
        rows = []

    nodes = [Node(id=user, label=user, size=22, color="#378ADD")]
    edges = []
    for r in rows:
        unusual = r["other_users"] == 0
        nodes.append(
            Node(id=r["pc"], label=r["pc"], size=12, color="#D85A30" if unusual else "#9c9a92")
        )
        edges.append(Edge(source=user, target=r["pc"]))
    if nodes:
        agraph(nodes=nodes, edges=edges, config=Config(width=700, height=450, directed=False))
    st.caption("Orange PCs are used by no one else in the sample.")
