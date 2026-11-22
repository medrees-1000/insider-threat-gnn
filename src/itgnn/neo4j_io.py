"""Write the user/PC graph and model scores to Neo4j, and query neighborhoods for the UI."""
import os
from pathlib import Path

import pandas as pd

from itgnn.data import load_config, prepare
from itgnn.graph import logon_edges


def get_driver():
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "changeme123"))
    return GraphDatabase.driver(uri, auth=auth)


def load_graph(driver, users, labels, edges):
    """Create User and PC nodes and LOGGED_ON relationships."""
    user_rows = [{"id": u, "malicious": u in labels} for u in users]
    edge_rows = [
        {"user": u, "pc": p, "count": int(c)} for u, p, c in edges.itertuples(index=False)
    ]
    with driver.session() as s:
        s.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
        s.run("CREATE CONSTRAINT pc_id IF NOT EXISTS FOR (p:PC) REQUIRE p.id IS UNIQUE")
        s.run(
            "UNWIND $rows AS r MERGE (u:User {id: r.id}) SET u.malicious = r.malicious",
            rows=user_rows,
        )
        s.run(
            "UNWIND $rows AS r "
            "MERGE (u:User {id: r.user}) MERGE (p:PC {id: r.pc}) "
            "MERGE (u)-[e:LOGGED_ON]->(p) SET e.count = r.count",
            rows=edge_rows,
        )


def write_scores(driver, scores):
    """scores: DataFrame indexed by user with gnn_score and/or baseline_score columns."""
    rows = [{"id": u, **{c: float(r[c]) for c in scores.columns if c.endswith("_score")}}
            for u, r in scores.iterrows()]
    with driver.session() as s:
        s.run("UNWIND $rows AS r MATCH (u:User {id: r.id}) SET u += r", rows=rows)


def neighborhood(driver, user_id, limit=40):
    """PCs this user logs on to, with how many other users share each PC.

    other_users == 0 means the PC is used by nobody else, a simple 'unusual access' signal.
    """
    query = (
        "MATCH (u:User {id: $id})-[e:LOGGED_ON]->(p:PC) "
        "OPTIONAL MATCH (p)<-[:LOGGED_ON]-(o:User) WHERE o.id <> $id "
        "RETURN p.id AS pc, e.count AS count, count(DISTINCT o) AS other_users "
        "ORDER BY count DESC LIMIT $limit"
    )
    with driver.session() as s:
        return [r.data() for r in s.run(query, id=user_id, limit=limit)]


def main(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    d = prepare(cfg)
    results = Path(cfg["results_dir"])
    scores = pd.read_csv(results / "gnn_scores.csv", index_col="user")[["gnn_score"]]
    base = results / "baseline_scores.csv"
    if base.exists():
        scores = scores.join(pd.read_csv(base, index_col="user")[["baseline_score"]])

    driver = get_driver()
    load_graph(driver, d["users"], d["labels"], logon_edges(d["logon"], d["users"]))
    write_scores(driver, scores)
    driver.close()
    print("Loaded graph and scores into Neo4j")


if __name__ == "__main__":
    main()
