// Threat map glue for Streamlit Components v2. The vendored Cytoscape.js source is prepended by app/threat_map.py.
// Everything is client-side: week slider, play, search, filters, focus and the details panel need no Streamlit rerun.
// Only the "Open in Investigate" button sends a trigger value to Python.
export default function (component) {
  const { data, setTriggerValue, parentElement } = component;
  const P = data.payload, T = data.theme, M = P.meta;
  const root = document.createElement('div'); root.className = 'tm'; parentElement.appendChild(root);
  const css = (o) => Object.entries(o).forEach(([k, v]) => root.style.setProperty('--' + k, v));
  css({ bg: T.page, panel: T.panel, canvas: T.canvas, border: T.border, text: T.text, text2: T.text2, muted: T.muted, accent: T.accent, ring: T.ring });
  const saved = (window.__tmState = window.__tmState || { view: 'week', week: null, minLvl: 0, sites: true, pcs: true, contacts: true, labels: true, leaves: false });
  const S = saved;
  const cal = M.calendar, calIdx = Object.fromEntries(cal.map((c, i) => [c.cw, i]));
  if (S.week === null || !(S.week in calIdx)) {  // open on the calendar week with the most flagged user-weeks
    const cnt = {}; P.uw.forEach((r) => (cnt[r.cw] = (cnt[r.cw] || 0) + 1));
    S.week = +Object.entries(cnt).sort((a, b) => b[1] - a[1])[0][0];
  }
  const LV = ['top 2%', 'top 1%', 'top 0.5%', 'top 0.25%'];
  const catNames = { leak: 'leak site', download: 'keylogger site', job: 'job site', cloud: 'cloud storage', competitor: 'competitor' };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const zr = (() => { const z = P.uw.map((r) => r.z); return [Math.min(...z), Math.max(...z)]; })();
  const usize = (z) => 20 + 28 * (z - zr[0]) / Math.max(zr[1] - zr[0], 1e-6);

  root.innerHTML = `
  <div class="bar">
    <div class="seg"><button data-view="week">Week view</button><button data-view="campaign">Campaign view</button></div>
    <div class="grp wk"><button class="ico" data-act="prev" title="Previous week">&#9664;</button>
      <input type="range" class="slider" min="0" max="${cal.length - 1}" step="1"><button class="ico" data-act="next" title="Next week">&#9654;</button>
      <button class="play" data-act="play">&#9654; Play</button><span class="wlabel"></span></div>
    <div class="grp cp"><span class="cinfo"></span></div>
    <div class="grp"><input class="search" list="tm-users" placeholder="Find a user…"><datalist id="tm-users"></datalist></div>
    <div class="grp"><label>Min risk <select class="minlvl">${LV.map((l, i) => `<option value="${i}">${l}</option>`).join('')}</select></label></div>
    <div class="grp tog"><label><input type="checkbox" data-t="sites"> Websites</label><label><input type="checkbox" data-t="pcs"> PCs</label>
      <label><input type="checkbox" data-t="contacts"> Contacts</label><label><input type="checkbox" data-t="labels"> Labels</label><label title="Show websites and contacts that only one flagged user touched"><input type="checkbox" data-t="leaves"> All neighbours</label>
      <button class="ico" data-act="fit" title="Fit to screen">Fit</button></div>
  </div>
  <div class="body"><div class="canvas"><div class="tip"></div><div class="empty"></div></div><div class="panel"></div></div>
  <div class="legend"></div>`;
  const $ = (q) => root.querySelector(q), $$ = (q) => [...root.querySelectorAll(q)];
  const canvas = $('.canvas'), tip = $('.tip'), panel = $('.panel');
  const evalOn = !!M.evaluation;

  const style = [
    { selector: 'node', style: { 'font-family': 'system-ui, sans-serif', 'font-size': 10, color: T.text, 'text-outline-width': 3, 'text-outline-color': T.canvas, 'text-valign': 'bottom', 'text-margin-y': 3, 'min-zoomed-font-size': 7, 'text-opacity': 0 } },
    { selector: 'node.labeled', style: { label: 'data(label)', 'text-opacity': 1, 'min-zoomed-font-size': 9 } },
    { selector: 'node.user.labeled, node.cat.labeled', style: { 'min-zoomed-font-size': 5 } },
    { selector: 'node.user', style: { shape: 'ellipse', width: 'data(size)', height: 'data(size)', 'background-color': 'data(color)', 'border-width': 1.5, 'border-color': T.outline, 'font-weight': 600, 'font-size': 12, 'z-index': 10 } },
    { selector: 'node.site', style: { shape: 'diamond', width: 15, height: 15, 'background-color': T.site } },
    { selector: 'node.site.cat', style: { width: 22, height: 22, 'background-color': 'data(color)', 'border-width': 1, 'border-color': T.ring, 'font-weight': 600 } },
    { selector: 'node.site.cat.neutral', style: { 'border-style': 'dashed', 'border-width': 2 } },
    { selector: 'node.pc', style: { shape: 'rectangle', width: 13, height: 13, 'background-color': T.pc } },
    { selector: 'node.pc.solo', style: { 'border-width': 2, 'border-color': T.risk[1] } },
    { selector: 'node.contact', style: { shape: 'ellipse', width: 8, height: 8, 'background-color': T.contact } },
    { selector: 'node.cluster', style: { shape: 'round-rectangle', 'background-opacity': 0.06, 'background-color': T.accent, 'border-width': 1, 'border-style': 'dashed', 'border-color': T.muted, label: 'data(label)', 'text-opacity': 1, 'text-valign': 'top', 'text-halign': 'center', 'font-size': 12, color: T.text2, 'text-outline-width': 0, padding: 16 } },
    { selector: 'node.ins', style: { 'border-width': 4, 'border-color': T.ring, 'border-style': 'double' } },
    { selector: 'edge', style: { width: 'data(w)', 'line-color': T.edge.dom, opacity: 0.55, 'curve-style': 'haystack', 'haystack-radius': 0 } },
    { selector: 'edge.pc', style: { 'line-color': T.edge.pc, 'line-style': 'dashed' } },
    { selector: 'edge.dom', style: { 'line-color': T.edge.dom, 'line-style': 'solid' } },
    { selector: 'edge.mail', style: { 'line-color': T.edge.mail, 'line-style': 'dotted' } },
    { selector: 'edge.mailu', style: { 'line-color': T.edge.mail, 'line-style': 'solid', opacity: 0.9, 'curve-style': 'bezier' } },
    { selector: 'node.lblh', style: { 'text-opacity': 0 } },
    { selector: 'node.lblforce', style: { 'text-opacity': 1, 'z-index': 999 } },
    { selector: '.dim', style: { opacity: 0.1, 'text-opacity': 0 } },
    { selector: 'edge.hl', style: { opacity: 0.95 } },
    { selector: 'node.sel', style: { 'border-width': 4, 'border-color': T.accent, 'border-style': 'solid' } },
  ];
  const cy = cytoscape({ container: canvas, elements: [], style, minZoom: 0.15, maxZoom: 4, textureOnViewport: true, hideEdgesOnViewport: true });
  canvas.style.position = 'relative';
  window.__tmDebug = { cy, root, setWeek: (w) => { S.week = w; draw(); }, setView: (v) => { S.view = v; draw(); }, focus: (id) => focus(cy.getElementById(id)), state: S };

  // ---- element builders ------------------------------------------------------------------------
  const riskColor = (lvl) => T.risk[lvl];
  const spiral = (i) => { const a = i * 2.399963, r = 12 * Math.sqrt(i + 1); return { x: r * Math.cos(a), y: r * Math.sin(a) }; };
  const catColor = (c) => T.cat[c] || T.site;

  function weekElements() {
    const els = [], users = P.uw.filter((r) => r.cw === S.week && r.lvl >= S.minLvl), ids = new Set(users.map((r) => r.i));
    const deg = {};
    const usable = [];
    P.edges.forEach((e) => {
      if (!ids.has(e[0])) return;
      const t = e[2];
      if (t === 3) { if (ids.has(e[1])) usable.push(e); return; }
      const c = P.conn[e[1]];
      if (c.k === 'domain' && !S.sites && !c.cat) return;
      if (c.k === 'pc' && !S.pcs) return;
      if (c.k === 'contact' && !S.contacts) return;
      usable.push(e); deg[e[1]] = (deg[e[1]] || 0) + 1;
    });
    let hidden = 0;
    const visible = usable.filter((e) => { if (e[2] === 3) return true; const c = P.conn[e[1]]; const ok = S.leaves || c.cat || c.k === 'pc' || deg[e[1]] >= 2; if (!ok) hidden++; return ok; });
    window.__tmDebug && (window.__tmDebug.hidden = hidden);
    users.forEach((r, n) => els.push({ group: 'nodes', data: { id: 'u' + r.i, label: r.u, size: usize(r.z), color: riskColor(r.lvl), kind: 'user', i: r.i }, classes: 'user labeled' + (r.ins ? ' ins' : ''), position: spiral(n) }));
    const made = new Set(); let n = users.length;
    visible.forEach((e, k) => {
      if (e[2] === 3) { els.push({ group: 'edges', data: { id: 'e' + k, source: 'u' + e[0], target: 'u' + e[1], w: 1.5 + Math.min(Math.log1p(e[3]), 3) }, classes: 'mailu' }); return; }
      const c = P.conn[e[1]], id = 'c' + e[1], shared = deg[e[1]] >= 2;
      if (!made.has(id)) {
        made.add(id);
        const cls = c.k === 'domain' ? 'site' + (c.cat ? ' cat' + (c.cat === 'competitor' ? ' neutral' : '') : '') : c.k;
        const lab = (S.labels && (c.cat || shared)) ? ' labeled' : (c.cat ? ' labeled' : '');
        els.push({ group: 'nodes', data: { id, label: c.cat ? `${c.n} · ${catNames[c.cat]}` : c.n, kind: c.k, ci: e[1], color: c.cat ? catColor(c.cat) : T.site }, classes: cls + (shared ? ' shared' : '') + lab + (c.k === 'pc' && e[4] === 0 ? ' solo' : ''), position: spiral(n++) });
      }
      els.push({ group: 'edges', data: { id: 'e' + k, source: 'u' + e[0], target: id, w: 0.8 + Math.min(Math.log1p(e[3]) * 0.7, 3.2) }, classes: e[2] === 0 ? 'pc' : e[2] === 1 ? 'dom' : 'mail' });
    });
    return els;
  }

  function campaignElements() {
    const C = P.campaign, els = [];
    const users = C.users.filter((u) => u.lvl >= S.minLvl);
    const uset = new Set(users.map((u) => C.users.indexOf(u)));
    C.clusters.forEach((sz, k) => { if (users.some((u) => u.cl === k)) els.push({ group: 'nodes', data: { id: 'k' + k, label: `Cluster ${k + 1} · ${sz} users` }, classes: 'cluster' }); });
    users.forEach((u, n) => {
      const idx = C.users.indexOf(u);
      const d = { id: 'u' + idx, label: u.u, size: usize(u.z) * (0.8 + Math.min(u.wk, 40) / 60), color: riskColor(u.lvl), kind: 'user', idx };
      if (u.cl >= 0) d.parent = 'k' + u.cl;
      els.push({ group: 'nodes', data: d, classes: 'user' + (S.labels && (S.leaves || u.lvl >= 2) ? ' labeled' : '') + (u.ins ? ' ins' : ''), position: { x: u.x, y: u.y } });
    });
    const used = new Set(), keepEdge = [];
    C.edges.forEach((e, k) => {
      if (!uset.has(e[0])) return;
      if (e[2] === 3) { if (uset.has(e[1])) keepEdge.push([e, k]); return; }
      const c = C.conn[e[1]];
      if ((c.k === 'domain' && !S.sites && !c.cat) || (c.k === 'pc' && !S.pcs) || (c.k === 'contact' && !S.contacts)) return;
      used.add(e[1]); keepEdge.push([e, k]);
    });
    const rank = [...used].sort((a, b) => C.conn[b].deg - C.conn[a].deg), top = new Set(rank.filter((j, k) => S.leaves || k < C.top_connectors || C.conn[j].cat));
    [...used].filter((j) => top.has(j)).forEach((j) => { const c = C.conn[j]; els.push({ group: 'nodes', data: { id: 'c' + j, label: c.cat ? `${c.n} · ${catNames[c.cat]}` : c.n, kind: c.k, ci: j, color: c.cat ? catColor(c.cat) : T.site },
      classes: (c.k === 'domain' ? 'site' + (c.cat ? ' cat' + (c.cat === 'competitor' ? ' neutral' : '') : '') : c.k) + (c.cat ? ' labeled' : ''), position: { x: c.x, y: c.y } }); });
    const perUser = {};   // default: each user's five strongest website/PC/contact links and three strongest user-to-user email links; "All neighbours" draws them all
    const strongest = keepEdge.filter(([e]) => e[2] === 3 || top.has(e[1])).sort((a, b) => b[0][3] - a[0][3]).filter(([e]) => { if (S.leaves) return true; const key = (e[2] === 3 ? 'm' : 'c') + e[0]; perUser[key] = (perUser[key] || 0) + 1; return perUser[key] <= (e[2] === 3 ? 3 : 5); });
    strongest.forEach(([e, k]) => els.push({ group: 'edges', data: { id: 'e' + k, source: 'u' + e[0], target: e[2] === 3 ? 'u' + e[1] : 'c' + e[1], w: 0.7 + Math.min(Math.log1p(e[3]) * 0.5, 3) },
      classes: e[2] === 0 ? 'pc' : e[2] === 1 ? 'dom' : e[2] === 2 ? 'mail' : 'mailu' }));
    return els;
  }

  // ---- label priority: draw labels in priority order and hide any that would overlap a higher-priority one ----------
  const prio = (n, nb) => (nb && nb.has(n.id()) ? 0 : n.hasClass('user') ? 1 : n.hasClass('cat') ? 2 : n.hasClass('shared') ? 3 : 4);
  const rectOf = (n) => { const bb = n.renderedBoundingBox({ includeLabels: true }), body = n.renderedBoundingBox({ includeLabels: false });
    return { x1: bb.x1 - 2, x2: bb.x2 + 2, y1: Math.min(body.y2, bb.y2 - 1), y2: bb.y2 + 2 }; };
  const hits = (a, b) => a.x1 < b.x2 && a.x2 > b.x1 && a.y1 < b.y2 && a.y2 > b.y1;
  function layoutLabels() {
    const nb = S.focusId ? new Set(cy.getElementById(S.focusId).closedNeighborhood('node').map((n) => n.id())) : null;
    const nodes = cy.nodes('.labeled').not('.cluster').toArray().sort((a, b) => prio(a, nb) - prio(b, nb) || (b.data('size') || 0) - (a.data('size') || 0));
    const bodies = cy.nodes('.user').map((n) => n.renderedBoundingBox({ includeLabels: false })), placed = [];
    cy.batch(() => nodes.forEach((n) => {
      const r = rectOf(n);
      const hit = placed.some((p) => hits(r, p)) || (!n.hasClass('user') && bodies.some((b) => hits(r, b)));
      n.toggleClass('lblh', hit); if (!hit) placed.push(r);
    }));
  }
  let lblTimer = null;
  const scheduleLabels = () => { if (lblTimer) clearTimeout(lblTimer); lblTimer = setTimeout(() => { lblTimer = null; layoutLabels(); }, 50); };
  cy.on('zoom pan dragfree', scheduleLabels);
  window.__tmDebug.labelStats = () => { const vis = cy.nodes('.labeled').not('.lblh').not('.dim').not('.cluster'); const rs = vis.map(rectOf); let ov = 0;
    for (let i = 0; i < rs.length; i++) for (let j = i + 1; j < rs.length; j++) if (hits(rs[i], rs[j])) ov++;
    return { labeled: cy.nodes('.labeled').not('.cluster').length, visible: vis.length, hidden: cy.nodes('.labeled.lblh').length, usersHidden: cy.nodes('.user.labeled.lblh').length, users: cy.nodes('.user').length, overlappingVisiblePairs: ov }; };

  // ---- draw / layout ---------------------------------------------------------------------------
  let playTimer = null, lastLayoutMs = 0;
  function draw() {
    cy.elements().remove(); cy.add(S.view === 'week' ? weekElements() : campaignElements());
    const t = performance.now();
    const n = cy.nodes().length;
    $('.empty').textContent = n ? '' : 'No flagged user-weeks match these filters.';
    if (n && S.view === 'campaign') cy.layout({ name: 'preset', fit: true, padding: 40 }).run();
    else if (n) cy.layout({ name: 'cose', animate: false, randomize: false, fit: true, padding: 36, nodeRepulsion: () => (S.view === 'week' ? 36000 : 16000), idealEdgeLength: () => (S.view === 'week' ? 120 : 70),
      edgeElasticity: () => 80, gravity: S.view === 'week' ? 0.09 : 0.3, numIter: S.view === 'week' ? 800 : 900, nestingFactor: 0.9, componentSpacing: 60 }).run();
    lastLayoutMs = Math.round(performance.now() - t);
    window.__tmDebug.lastLayoutMs = lastLayoutMs; window.__tmDebug.nodes = n; window.__tmDebug.edges = cy.edges().length;
    syncControls(); S.focusId = null; renderLegend(); layoutLabels(); showSummary();
  }

  function syncControls() {
    $$('.seg button').forEach((b) => b.classList.toggle('on', b.dataset.view === S.view));
    $('.wk').style.display = S.view === 'week' ? '' : 'none'; $('.cp').style.display = S.view === 'campaign' ? '' : 'none';
    $('.slider').value = calIdx[S.week]; $('.wlabel').textContent = `calendar week ${S.week} · ${cal[calIdx[S.week]].d}`;
    $('.minlvl').value = S.minLvl; $$('[data-t]').forEach((c) => (c.checked = S[c.dataset.t]));
    $('.play').innerHTML = playTimer ? '&#10074;&#10074; Pause' : '&#9654; Play';
    const C = P.campaign; $('.cinfo').textContent = `${C.users.length} flagged users · ${C.clusters.length} communities (Louvain, seed ${C.louvain_seed}, modularity ${C.modularity})`;
  }

  // ---- details panel ---------------------------------------------------------------------------
  const bars = (c, groups) => {
    const m = Math.max(...c.map(Math.abs), 1e-6);
    return c.map((v, k) => ({ v, name: groups[k] })).sort((a, b) => b.v - a.v).map((o) => `<div class="brow"><span class="bn">${esc(o.name)}</span><span class="bt"><span class="bf ${o.v >= 0 ? 'pos' : 'neg'}" style="width:${Math.max(2, 100 * Math.abs(o.v) / m)}%"></span></span><span class="bv">${o.v >= 0 ? '+' : ''}${o.v.toFixed(2)}</span></div>`).join('');
  };
  const chip = (c) => `<span class="chip" style="--c:${catColor(c)}">${esc(catNames[c] || c)}</span>`;
  const lvlBadge = (l) => `<span class="badge" style="--c:${riskColor(l)}">${LV[l]}</span>`;
  const openBtn = (user, week) => `<button class="open" data-user="${esc(user)}" data-week="${esc(week)}">Open in Investigate &rarr;</button>`;
  function showSummary() {
    const n = cy.nodes('.user').length;
    panel.innerHTML = `<h3>${S.view === 'week' ? 'Flagged user-weeks' : 'Campaign view'}</h3><p class="sub">${S.view === 'week' ? `${n} flagged user-weeks in calendar week ${S.week} (${cal[calIdx[S.week]].d}); ${cy.nodes().length - n} neighbours shown` + (window.__tmDebug.hidden ? `, ${window.__tmDebug.hidden} links to unshared websites/contacts hidden (tick "All neighbours" to show them).` : '.') : `${n} flagged users linked through websites, PCs and email shared by at least two of them; communities are found with Louvain (weak structure when modularity is low).`}</p>
      <p class="hint">Click a node to focus it and see the evidence. Drag nodes to rearrange. Scroll to zoom.</p>` + (() => { const st = window.__tmDebug.labelStats(); return st.hidden ? `<p class="hint">${st.hidden} labels are hidden to keep the text readable (${st.usersHidden} of ${st.users} people). Zoom in, hover, or click a dot to see its name.</p>` : ''; })() + (evalOn ? `<p class="hint ev">EVALUATION VIEW: a double ring marks true insiders. Analysts never see this.</p>` : '');
  }
  function userPanel(r, ele) {
    const nb = ele.neighborhood('node'), sites = nb.filter((n) => n.hasClass('site')), cats = sites.filter((n) => n.hasClass('cat')), pcs = nb.filter((n) => n.hasClass('pc')), con = nb.filter((n) => n.hasClass('contact')), usr = nb.filter((n) => n.hasClass('user'));
    const edgeW = (n) => { const e = ele.edgesWith(n)[0]; return e ? e.data('w') : 0; };
    const rec = P.uw[r];
    const catRows = cats.map((n) => `<li>${chip(P.conn[n.data('ci')].cat)} <b>${esc(P.conn[n.data('ci')].n)}</b></li>`).join('') || '<li class="none">no named-site visits in this week</li>';
    const pcRows = pcs.map((n) => `<li>${esc(P.conn[n.data('ci')].n)} ${n.hasClass('solo') ? '<span class="none">(nobody else that week)</span>' : '<span class="none">(shared with others that week)</span>'}${[...n.neighborhood('node.user')].filter((u) => u.id() !== ele.id()).map((u) => ` <span class="peer">${esc(u.data('label'))}</span>`).join('')}</li>`).join('') || '<li class="none">no PCs shown</li>';
    panel.innerHTML = `<h3>${esc(rec.u)} <span class="wk">${rec.d}</span></h3><div>${lvlBadge(rec.lvl)} <span class="sub">risk percentile ${rec.p.toFixed(2)}</span></div>
      <table class="kv"><tr><td>Score</td><td>${rec.s.toFixed(5)} <span class="none">&plusmn; ${rec.sd.toFixed(4)} over 5 repeats</span></td></tr><tr><td>Log-odds</td><td>${rec.z.toFixed(2)}</td></tr><tr><td>Generic-feature score</td><td>${rec.g.toFixed(5)} <span class="none">(no named-site features)</span></td></tr></table>
      ${openBtn(rec.u, rec.d)}
      <h4>Why it scored (log-odds)</h4>${bars(rec.c, M.groups)}
      <h4>Named-site evidence</h4><ul>${catRows}</ul>
      <h4>PCs</h4><ul>${pcRows}</ul>
      <h4>Email</h4><ul><li>${con.length} internal contacts shown${usr.length ? `, ${usr.length} other flagged user(s) emailed in this week: ${usr.map((u) => `<span class="peer">${esc(u.data('label'))}</span>`).join(' ')}` : ''}</li><li>${rec.ne} external recipients</li></ul>`;
  }
  function connPanel(ele) {
    const c = (S.view === 'week' ? P.conn : P.campaign.conn)[ele.data('ci')];
    const us = ele.neighborhood('node.user').sort((a, b) => b.data('size') - a.data('size'));
    const rows = us.map((u) => { const rec = S.view === 'week' ? P.uw[u.data('i')] : P.campaign.users[u.data('idx')]; return `<li><b>${esc(rec.u)}</b> ${lvlBadge(rec.lvl)}</li>`; }).join('');
    panel.innerHTML = `<h3>${esc(c.n)}</h3><div>${c.cat ? chip(c.cat) + ' <span class="none">named-site category (also an XGBoost feature; evidence for the analyst)</span>' : `<span class="none">${c.k === 'pc' ? 'PC' : c.k === 'domain' ? 'website' : 'email contact'}</span>`}</div>
      <h4>${us.length} flagged user${us.length === 1 ? '' : 's'} linked here${S.view === 'week' ? ' this week' : ''}</h4><ul>${rows}</ul>`;
  }
  function campUserPanel(ele) {
    const u = P.campaign.users[ele.data('idx')], C = P.campaign;
    const cons = ele.connectedEdges().map((e) => { const o = e.source().id() === ele.id() ? e.target() : e.source(); return { o, w: e.data('w') }; }).filter((x) => x.o.data('ci') !== undefined).sort((a, b) => b.w - a.w).slice(0, 10);
    panel.innerHTML = `<h3>${esc(u.u)}</h3><div>${lvlBadge(u.lvl)} <span class="sub">best week ${u.bd} · percentile ${u.p.toFixed(2)}</span></div>
      <table class="kv"><tr><td>Flagged weeks</td><td>${u.wk}</td></tr><tr><td>Best score</td><td>${u.s.toFixed(5)}</td></tr><tr><td>Community</td><td>${u.cl >= 0 ? 'Cluster ' + (u.cl + 1) + ' (' + C.clusters[u.cl] + ' users)' : 'other'}</td></tr></table>
      ${openBtn(u.u, u.bd)}
      <h4>Strongest shared links</h4><ul>${cons.map((x) => { const c = C.conn[x.o.data('ci')]; return `<li>${c.cat ? chip(c.cat) + ' ' : ''}${esc(c.n)} <span class="none">${c.k === 'domain' ? 'website' : c.k === 'pc' ? 'PC' : 'contact'} · ${c.deg} flagged users</span></li>`; }).join('') || '<li class="none">none</li>'}</ul>`;
  }
  function focus(ele) {
    cy.elements().removeClass('dim hl sel'); cy.nodes('.tmplbl').removeClass('labeled tmplbl lblh');
    if (!ele || ele.empty() || ele.isEdge()) { S.focusId = null; showSummary(); scheduleLabels(); return; }
    const nb = ele.closedNeighborhood(); cy.elements().not(nb).addClass('dim'); nb.edges().addClass('hl'); ele.addClass('sel');
    S.focusId = ele.id(); nb.nodes().not('.labeled').not('.cluster').addClass('labeled tmplbl'); scheduleLabels();
    if (ele.data('kind') === 'user') (S.view === 'week' ? userPanel(ele.data('i'), ele) : campUserPanel(ele)); else connPanel(ele);
  }
  cy.on('tap', 'node', (e) => { if (!e.target.hasClass('cluster')) focus(e.target); });
  cy.on('tap', (e) => { if (e.target === cy) focus(null); });
  panel.addEventListener('click', (e) => { const b = e.target.closest('.open'); if (b) setTriggerValue('open', { user: b.dataset.user, week: b.dataset.week, ts: Date.now() }); });

  // ---- hover tooltip ---------------------------------------------------------------------------
  cy.on('mouseover', 'node', (e) => {
    const n = e.target; if (n.hasClass('cluster')) return;
    let h;
    if (n.data('kind') === 'user') { const r = S.view === 'week' ? P.uw[n.data('i')] : P.campaign.users[n.data('idx')]; h = `<b>${esc(r.u)}</b><br>${LV[r.lvl]} · percentile ${r.p.toFixed(2)}${S.view === 'campaign' ? `<br>${r.wk} flagged weeks` : '<br>' + r.d}`; }
    else { const c = (S.view === 'week' ? P.conn : P.campaign.conn)[n.data('ci')]; h = `<b>${esc(c.n)}</b><br>${c.cat ? esc(catNames[c.cat]) + ' (named-site category)' : c.k === 'pc' ? 'PC' : c.k === 'domain' ? 'website' : 'email contact'}`; }
    tip.innerHTML = h; tip.style.display = 'block'; n.addClass('labeled lblforce');
  });
  cy.on('mousemove', (e) => { if (e.originalEvent) { const r = canvas.getBoundingClientRect(); tip.style.left = Math.min(e.originalEvent.clientX - r.left + 14, r.width - 190) + 'px'; tip.style.top = (e.originalEvent.clientY - r.top + 14) + 'px'; } });
  cy.on('mouseout', 'node', (e) => { tip.style.display = 'none'; const n = e.target; n.removeClass('lblforce'); if (!n.hasClass('user') && !n.hasClass('cat') && !n.hasClass('shared') && !n.hasClass('tmplbl')) n.removeClass('labeled'); });

  // ---- controls --------------------------------------------------------------------------------
  const go = (d) => { const i = Math.min(Math.max(calIdx[S.week] + d, 0), cal.length - 1); S.week = cal[i].cw; draw(); };
  $$('.seg button').forEach((b) => (b.onclick = () => { S.view = b.dataset.view; stopPlay(); draw(); }));
  $('.slider').oninput = (e) => { S.week = cal[+e.target.value].cw; draw(); };
  $('[data-act=prev]').onclick = () => go(-1); $('[data-act=next]').onclick = () => go(1); $('[data-act=fit]').onclick = () => cy.fit(undefined, 36);
  const stopPlay = () => { if (playTimer) { clearInterval(playTimer); playTimer = null; } };
  $('[data-act=play]').onclick = () => { if (playTimer) { stopPlay(); syncControls(); return; } playTimer = setInterval(() => { if (calIdx[S.week] >= cal.length - 1) { stopPlay(); syncControls(); return; } go(1); }, 1400); syncControls(); };
  $('.minlvl').onchange = (e) => { S.minLvl = +e.target.value; draw(); };
  $$('[data-t]').forEach((c) => (c.onchange = () => { S[c.dataset.t] = c.checked; draw(); }));
  const users = [...new Set(P.uw.map((r) => r.u))].sort();
  $('#tm-users').innerHTML = users.map((u) => `<option value="${esc(u)}">`).join('');
  $('.search').onchange = (e) => {
    const u = e.target.value.trim(); if (!u) return;
    if (S.view === 'week') { const best = P.uw.filter((r) => r.u === u).sort((a, b) => b.s - a.s)[0]; if (!best) return; S.week = best.cw; S.minLvl = 0; draw(); focus(cy.getElementById('u' + best.i)); }
    else { const k = P.campaign.users.findIndex((x) => x.u === u); if (k >= 0) focus(cy.getElementById('u' + k)); }
  };

  function renderLegend() {
    const sw = (c, shape) => `<span class="sw ${shape}" style="--c:${c}"></span>`;
    $('.legend').innerHTML = `<span class="lg"><b>Risk</b> ${T.risk.map((c, i) => `${sw(c, 'dot')}${LV[i]}`).join(' ')}</span>
      <span class="lg">${sw(T.site, 'dia')}website ${sw(T.pc, 'sq')}PC ${sw(T.contact, 'dot sm')}email contact ${sw(T.pc, 'sq solo')}PC nobody else used</span>
      <span class="lg"><b>Named-site category</b> ${['leak', 'download', 'job', 'cloud'].map((c) => `${sw(catColor(c), 'dia')}${catNames[c]}`).join(' ')} ${sw(T.cat.competitor, 'dia dash')}competitor</span>
      <span class="lg">edges: <i class="e pc"></i>logged on <i class="e dom"></i>visited <i class="e mail"></i>emailed</span>${evalOn ? '<span class="lg ev">double ring = true insider (evaluation view)</span>' : ''}`;
  }

  draw();
  if (data.focus) { const f = P.uw.filter((r) => r.u === data.focus.user).sort((a, b) => Math.abs(new Date(a.d) - new Date(data.focus.week)) - Math.abs(new Date(b.d) - new Date(data.focus.week)))[0]; if (f) { S.week = f.cw; S.minLvl = 0; S.view = 'week'; draw(); focus(cy.getElementById('u' + f.i)); } }
  return () => { stopPlay(); try { cy.destroy(); } catch (e) {} };
}
