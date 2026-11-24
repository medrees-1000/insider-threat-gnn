"""Load the exported flagged-week neighbourhoods into Neo4j and query them for the UI.

Schema: (:User {id}) -[:HAS_WEEK]-> (:UserWeek {key, user, week_start, score, score_sd, score_generic, top_groups, ...})
-[:LOGGED_ON {week, logons, other_users}]-> (:PC {id}); -[:VISITED {week, visits}]-> (:Domain {name, category, category_note});
-[:EMAILED {week, mails}]-> (:User). No ground-truth label, scenario or attack window is ever written to Neo4j.
Credentials come from the environment or a gitignored `.env` (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD); no default password.
"""
import os
from pathlib import Path

import pandas as pd

from itgnn.ui_data import CATEGORY_NOTE, EDGE_COLS, normalise_rows, week_key

REL = {"LOGGED_ON": "logons", "VISITED": "visits", "EMAILED": "mails"}
WEEK_PROPS = ["user", "score", "score_sd", "score_generic", "cal_week", "top_groups", "n_external_recipients"]


def load_queries(path="neo4j/queries.cypher"):
    """{name: {"what": description, "cypher": query}} parsed from the annotated .cypher file ('// name:' starts a query)."""
    out, cur = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith("// name:"):
            cur = line.split(":", 1)[1].strip(); out[cur] = {"what": "", "cypher": ""}
        elif cur and line.startswith("// what:"):
            out[cur]["what"] = line.split(":", 1)[1].strip()
        elif cur and not line.startswith("//") and line.strip():
            out[cur]["cypher"] += line.strip() + " "
    return {k: {"what": v["what"], "cypher": v["cypher"].strip()} for k, v in out.items()}


def browser_link(cypher, base="http://localhost:7474", **params):
    """Neo4j Browser URL that pre-fills the editor with the query; `$name` placeholders are replaced by quoted literals."""
    from urllib.parse import quote

    for k, v in params.items():
        cypher = cypher.replace(f"${k}", "'" + str(v).replace("'", "\\'") + "'")
    return f"{base}/browser/?cmd=edit&arg={quote(cypher)}"


def load_env(path=".env"):
    """Read KEY=VALUE lines into os.environ without overriding variables that are already set."""
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def get_driver(env_path=".env"):
    from neo4j import GraphDatabase

    load_env(env_path)
    missing = [k for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"missing {missing}: copy .env.example to .env and set NEO4J_PASSWORD")
    return GraphDatabase.driver(os.environ["NEO4J_URI"], auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))


def week_rows(weeks):
    """UserWeek rows (plain python values) for UNWIND."""
    rows = []
    for r in weeks.itertuples(index=False):
        d = r._asdict()
        props = {c: (float(d[c]) if isinstance(d[c], (float, int)) and c not in ("user", "top_groups", "cal_week", "n_external_recipients") else d[c]) for c in WEEK_PROPS}
        props["cal_week"] = int(d["cal_week"]); props["n_external_recipients"] = int(d["n_external_recipients"]); props["week_start"] = pd.Timestamp(d["week_start"]).strftime("%Y-%m-%d")
        rows.append({"key": d["key"], "user": d["user"], "props": props})
    return rows


def edge_rows(edges, edge_type):
    e = edges[edges["edge_type"] == edge_type]
    return [{"key": week_key(r.user, r.week_start), "week": pd.Timestamp(r.week_start).strftime("%Y-%m-%d"), "other": r.other_id, "weight": float(r.weight),
             "category": r.category, "other_users": None if pd.isna(r.other_users) else float(r.other_users)} for r in e.itertuples(index=False)]


def batches(rows, n=5000):
    for i in range(0, len(rows), n):
        yield rows[i : i + n]


CYPHER = {
    "weeks": "UNWIND $rows AS r MERGE (u:User {id: r.user}) MERGE (w:UserWeek {key: r.key}) SET w += r.props MERGE (u)-[:HAS_WEEK]->(w)",
    "LOGGED_ON": "UNWIND $rows AS r MATCH (w:UserWeek {key: r.key}) MERGE (p:PC {id: r.other}) "
                 "MERGE (w)-[e:LOGGED_ON]->(p) SET e.week = r.week, e.logons = r.weight, e.other_users = r.other_users",
    "VISITED": "UNWIND $rows AS r MATCH (w:UserWeek {key: r.key}) MERGE (d:Domain {name: r.other}) "
               "SET d.category = CASE WHEN r.category = '' THEN null ELSE r.category END, "
               "d.category_note = CASE WHEN r.category = '' THEN null ELSE $note END "
               "MERGE (w)-[e:VISITED]->(d) SET e.week = r.week, e.visits = r.weight",
    "EMAILED": "UNWIND $rows AS r MATCH (w:UserWeek {key: r.key}) MERGE (v:User {id: r.other}) "
               "MERGE (w)-[e:EMAILED]->(v) SET e.week = r.week, e.mails = r.weight",
}
CONSTRAINTS = ["CREATE CONSTRAINT user_id IF NOT EXISTS FOR (n:User) REQUIRE n.id IS UNIQUE",
               "CREATE CONSTRAINT userweek_key IF NOT EXISTS FOR (n:UserWeek) REQUIRE n.key IS UNIQUE",
               "CREATE CONSTRAINT pc_id IF NOT EXISTS FOR (n:PC) REQUIRE n.id IS UNIQUE",
               "CREATE CONSTRAINT domain_name IF NOT EXISTS FOR (n:Domain) REQUIRE n.name IS UNIQUE"]


def load_ui_graph(driver, weeks, edges, batch=5000):
    """Idempotent (MERGE) load of the flagged-week neighbourhoods. Returns counts of what was sent."""
    sent = {"weeks": 0, "LOGGED_ON": 0, "VISITED": 0, "EMAILED": 0}
    with driver.session() as s:
        for c in CONSTRAINTS:
            s.run(c)
        for b in batches(week_rows(weeks), batch):
            s.run(CYPHER["weeks"], rows=b); sent["weeks"] += len(b)
        for t in ("LOGGED_ON", "VISITED", "EMAILED"):
            for b in batches(edge_rows(edges, t), batch):
                s.run(CYPHER[t], rows=b, note=CATEGORY_NOTE) if t == "VISITED" else s.run(CYPHER[t], rows=b)
                sent[t] += len(b)
    return sent


NEIGHBOURHOOD = (
    "MATCH (w:UserWeek {key: $key})-[e:LOGGED_ON|VISITED|EMAILED]->(o) "
    "RETURN type(e) AS edge_type, coalesce(o.id, o.name) AS other_id, labels(o)[0] AS other_kind, "
    "coalesce(e.logons, e.visits, e.mails) AS weight, o.category AS category, e.other_users AS other_users"
)


ALL_EDGES = (
    "MATCH (w:UserWeek)-[e:LOGGED_ON|VISITED|EMAILED]->(o) "
    "RETURN w.user AS user, w.week_start AS week_start, type(e) AS edge_type, coalesce(o.id, o.name) AS other_id, labels(o)[0] AS other_kind, "
    "coalesce(e.logons, e.visits, e.mails) AS weight, o.category AS category, e.other_users AS other_users"
)


class Neo4jSource:
    """Neighbourhood of a user-week from Neo4j (the preferred backend); returns the same frame as ParquetSource."""

    name = "neo4j"

    def __init__(self, driver):
        self.driver = driver

    def has_week(self, user, week_start):
        with self.driver.session() as s:
            return s.execute_read(lambda tx: tx.run("MATCH (w:UserWeek {key: $key}) RETURN count(w) AS n", key=week_key(user, week_start)).single()["n"]) > 0

    def all_edges(self):
        """Same table as ParquetSource.all_edges, from one Cypher query."""
        with self.driver.session() as s:
            rows = s.execute_read(lambda tx: [r.data() for r in tx.run(ALL_EDGES)])
        e = pd.DataFrame(rows, columns=["user", "week_start", "edge_type", "other_id", "other_kind", "weight", "category", "other_users"])
        e["week_start"] = pd.to_datetime(e["week_start"])
        e["weight"] = e["weight"].astype(float)
        e["category"] = e["category"].fillna("")
        e["other_users"] = e["other_users"].astype(float)
        return e.sort_values(["user", "week_start", "edge_type", "other_id"]).reset_index(drop=True)

    def neighbourhood(self, user, week_start):
        key = week_key(user, week_start)
        with self.driver.session() as s:
            rows = s.execute_read(lambda tx: [r.data() for r in tx.run(NEIGHBOURHOOD, key=key)])
        return normalise_rows(rows)


def auto_source(results="results", env_path=".env", timeout=3.0):
    """Neo4j when it is reachable AND loaded, otherwise the parquet fallback. Returns (source, note)."""
    from itgnn.ui_data import ParquetSource

    try:
        d = get_driver(env_path)
        d.verify_connectivity()
        with d.session() as s:
            n = s.run("MATCH (w:UserWeek) RETURN count(w) AS n").single()["n"]
        if n > 0:
            return Neo4jSource(d), f"Neo4j ({n:,} user-weeks loaded)"
        note = "Neo4j is reachable but empty (run `python -m itgnn.neo4j_ui load`); using the parquet fallback"
    except Exception as exc:  # not running, no credentials, bad password...
        note = f"Neo4j unavailable ({type(exc).__name__}); using the parquet fallback"
    return ParquetSource(results), note


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Load the flagged-week neighbourhoods into Neo4j")
    ap.add_argument("command", choices=["load"])
    ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    r = Path(a.results)
    d = get_driver()
    sent = load_ui_graph(d, pd.read_parquet(r / "ui_weeks.parquet"), pd.read_parquet(r / "ui_edges.parquet"))
    with d.session() as s:
        print({"sent": sent, "nodes": {l: s.run(f"MATCH (n:{l}) RETURN count(n) AS n").single()["n"] for l in ["User", "UserWeek", "PC", "Domain"]},
               "relationships": {t: s.run(f"MATCH ()-[e:{t}]->() RETURN count(e) AS n").single()["n"] for t in ["HAS_WEEK", "LOGGED_ON", "VISITED", "EMAILED"]}})
    d.close()


if __name__ == "__main__":
    main()
