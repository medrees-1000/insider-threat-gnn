// Queries for the loaded flagged-week graph (User, UserWeek, PC, Domain). Paste into Neo4j Browser (http://localhost:7474).
// Parameters are set with :param, for example   :param key => 'AAE0190|2010-07-12'
// The app's "Open in Neo4j Browser" links use the same queries with the values filled in (see src/itgnn/neo4j_ui.py: load_queries).

// name: neighbourhood
// what: everything one user-week is connected to (PCs, websites incl. named-site categories, internal email contacts)
MATCH (w:UserWeek {key: $key})-[e:LOGGED_ON|VISITED|EMAILED]->(o)
RETURN w, e, o;

// name: visitors
// what: all flagged user-weeks that visited one named-site domain (set $domain, e.g. 'wikileaks.org'), highest score first
MATCH (w:UserWeek)-[e:VISITED]->(d:Domain {name: $domain})
RETURN w, e, d
ORDER BY w.score DESC;

// name: shared_pc
// what: flagged users who used the same PC in the same week (set $week, e.g. '2010-07-12')
MATCH (a:UserWeek {week_start: $week})-[:LOGGED_ON]->(p:PC)<-[:LOGGED_ON]-(b:UserWeek {week_start: $week})
WHERE a.user < b.user
RETURN a, p, b;

// name: campaign
// what: pairs of flagged users linked through at least two shared websites or PCs (a campaign-style cluster), strongest first
MATCH (a:UserWeek)-[:VISITED|LOGGED_ON]->(x)<-[:VISITED|LOGGED_ON]-(b:UserWeek)
WHERE a.user < b.user
WITH a.user AS user_a, b.user AS user_b, collect(DISTINCT coalesce(x.name, x.id)) AS shared
WHERE size(shared) >= 2
RETURN user_a, user_b, size(shared) AS n_shared, shared[0..5] AS examples
ORDER BY n_shared DESC
LIMIT 50;
