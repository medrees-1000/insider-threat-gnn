"""Real-browser regression tests for the Threat map (headless Chromium, real mouse and wheel events; no AppTest).

Bug these guard against: Streamlit scrolls inside its own container, Cytoscape caches the canvas's client rectangle, so after the page was
scrolled every pointer event hit-tested a scroll-distance away from the node and hover, click and drag did nothing.

Opt-in because it starts a Streamlit server and a browser (about a minute):  ITGNN_BROWSER=1 pytest tests/browser -s
It needs results/ui_*.parquet (the real analyst data) and a Playwright Chromium in ~/.cache/ms-playwright.
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import cdp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    os.environ.get("ITGNN_BROWSER") != "1" or cdp.CH is None or not (ROOT / "results" / "ui_scores.parquet").exists(),
    reason="set ITGNN_BROWSER=1 (needs headless Chromium and results/ui_scores.parquet)")

SCROLL_PX = 500   # wheel distance scrolled before every interaction


@pytest.fixture(scope="module")
def app_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless", "true", "--server.port", str(port),
                             "--browser.gatherUsageStats", "false"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1).read() == b"ok": break
            except Exception: time.sleep(1)
        else:
            pytest.fail("streamlit did not start")
        yield f"http://localhost:{port}/"
    finally:
        proc.terminate(); proc.wait(timeout=20)


@pytest.fixture(scope="module")
def browser(app_url):
    b = cdp.Browser(width=1500, height=900)
    b.goto(app_url, "Finding insiders from everyday activity", 120); time.sleep(2)
    b.click_text("Threat map")
    for _ in range(90):                      # the map has mounted once Cytoscape has drawn nodes
        if b.eval("!!(window.__tmDebug && window.__tmDebug.nodes > 0)"): break
        time.sleep(1)
    else:
        pytest.fail("the Threat map did not draw")
    time.sleep(2)
    yield b
    b.close()


# The node's REAL client position is computed from a fresh getBoundingClientRect, never from Cytoscape's cache.
_PICK = """(() => { const d = window.__tmDebug, cy = d.cy, r = d.root.querySelector('.canvas').getBoundingClientRect();
  const all = cy.nodes().toArray().map((n) => ({ id: n.id(), user: n.hasClass('user'), x: r.left + n.renderedPosition().x, y: r.top + n.renderedPosition().y, s: n.renderedWidth() }));
  const ok = all.filter((u) => u.user && u.x > r.left + 30 && u.x < r.right - 30 && u.y > Math.max(r.top + 30, 90) && u.y < Math.min(r.bottom, innerHeight) - 30);
  const clear = ok.filter((u) => all.every((o) => o.id === u.id || Math.hypot(o.x - u.x, o.y - u.y) > 34));
  const c = (clear.length ? clear : ok).sort((a, b) => b.s - a.s)[0];
  const bb = cy.renderer().containerBB;
  return JSON.stringify(c ? { ...c, label: cy.getElementById(c.id).data('label'), scroll: document.querySelector('[data-testid=stMain]').scrollTop, zoom: cy.zoom(), top: r.top, cachedTop: bb ? bb[1] : null } : null); })()"""
_TIP = "(() => { const t = window.__tmDebug.root.querySelector('.tip'); return t.style.display === 'block' ? t.innerText : ''; })()"
_POS = "(() => { const p = window.__tmDebug.cy.getElementById('%s').position(); return [p.x, p.y]; })()"


def scroll_page(b):
    b.eval("document.querySelector('[data-testid=stMain]').scrollTo(0, 0)"); time.sleep(0.5)
    b.wheel(1300, 400, SCROLL_PX); time.sleep(0.8)          # real wheel event over the page margin right of the map, not over the canvas


def hover(b, node):
    b.mouse("mouseMoved", node["x"] + 90, node["y"] + 90); b.mouse("mouseMoved", node["x"], node["y"]); time.sleep(0.4)
    return b.eval(_TIP)


def scenario(b, view):
    """Switch view, scroll the page by SCROLL_PX with the wheel, then hover, click and drag a person node. Returns what happened."""
    b.eval(f"window.__tmDebug.root.querySelector('[data-view={view}]').click()"); time.sleep(1.5)
    scroll_page(b)
    node = json.loads(b.eval(_PICK)); assert node, "no clear person node in view"
    out = {"view": view, "node": node, "tip": hover(b, node)}
    b.click(node["x"], node["y"]); time.sleep(0.5)                                                # click-to-focus
    out["focus"] = b.eval("window.__tmDebug.state.focusId"); out["panel"] = b.eval("window.__tmDebug.root.querySelector('.panel h3').innerText")
    before = b.eval(_POS % node["id"])
    b.drag(node["x"], node["y"], node["x"] + 60, node["y"] + 40); time.sleep(0.8)                 # drag by (+60, +40) screen pixels
    after = b.eval(_POS % node["id"])
    out["moved_px"] = [round((after[0] - before[0]) * node["zoom"], 1), round((after[1] - before[1]) * node["zoom"], 1)]
    return out


@pytest.mark.parametrize("view", ["week", "campaign"])
def test_pointer_events_after_page_scroll(browser, view):
    r = scenario(browser, view); print(json.dumps(r, indent=1))
    node = r["node"]
    assert node["scroll"] >= SCROLL_PX * 0.8, f"the page did not scroll ({node['scroll']} px)"
    assert node["label"] in r["tip"], f"hover tooltip missing after a {node['scroll']} px scroll: {r['tip']!r}"
    assert r["focus"] == node["id"] and node["label"] in r["panel"], f"click did not focus/fill the panel: {r['focus']!r} {r['panel']!r}"
    assert abs(r["moved_px"][0] - 60) < 8 and abs(r["moved_px"][1] - 40) < 8, f"drag moved the node {r['moved_px']} screen px, expected about [60, 40]"


def test_pointer_events_after_week_change_and_filters(browser):
    """Re-render paths (next week, a filter, back to the week view) must keep hit-testing correct after a scroll."""
    browser.eval("window.__tmDebug.root.querySelector('[data-view=week]').click()"); time.sleep(1)
    browser.eval("window.__tmDebug.root.querySelector('[data-act=next]').click()"); time.sleep(1.5)
    browser.eval("(() => { const s = window.__tmDebug.root.querySelector('.minlvl'); s.value = '0'; s.dispatchEvent(new Event('change')); })()"); time.sleep(1.5)
    scroll_page(browser)
    node = json.loads(browser.eval(_PICK))
    assert node["label"] in hover(browser, node)
