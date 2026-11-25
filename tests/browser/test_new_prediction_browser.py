"""Real-browser test for hypothetical workers on the Threat map (opt-in: ITGNN_BROWSER=1 pytest tests/browser -s).
Adds a worker in the New prediction tab with a real mouse, opens it on the map, then scrolls 500 px and hovers, clicks and drags its node."""
import json
import time

import pytest

import test_threat_map_browser as base   # reuses the opt-in skip, the app server fixture and the scroll/hover helpers
from test_threat_map_browser import app_url, cdp, SCROLL_PX, scroll_page, hover, _POS  # noqa: F401

pytestmark = base.pytestmark

_BTN = """(() => { const e = [...document.querySelectorAll('button')].find((b) => b.innerText.trim() === %s && !b.disabled); if (!e) return null;
  e.scrollIntoView({block: 'center'}); const r = e.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; })()"""


def press(b, text):
    b.eval(_BTN % json.dumps(text)); time.sleep(0.6)
    xy = b.eval(_BTN % json.dumps(text)); assert xy, f"button {text!r} not found or disabled"
    b.click(*xy); time.sleep(2.5)


def type_into(b, label, text):
    js = """(() => { const el = [...document.querySelectorAll('input')].find((e) => (e.getAttribute('aria-label') || '').startsWith(%s)); el.focus(); document.execCommand('insertText', false, %s); return true; })()"""
    assert b.eval(js % (json.dumps(label), json.dumps(text)))


def pick(b, label, text):
    """Open a multiselect with a real click, type, and click the first option (typing via execCommand does not open Streamlit's dropdown)."""
    js = """(() => { const el = [...document.querySelectorAll('input')].find((e) => (e.getAttribute('aria-label') || '').startsWith(%s)); el.scrollIntoView({block: 'center'}); const r = el.getBoundingClientRect(); return [r.x + 20, r.y + r.height / 2]; })()"""
    b.eval(js % json.dumps(label)); time.sleep(0.5)
    b.click(*b.eval(js % json.dumps(label))); time.sleep(0.5); b.send("Input.insertText", text=text); time.sleep(1.5)
    o = b.eval("(() => { const e = document.querySelector('[role=option]'); if (!e) return null; const r = e.getBoundingClientRect(); return [r.x + 30, r.y + r.height / 2]; })()")
    assert o, "no option offered"
    b.click(*o); time.sleep(2)


@pytest.fixture(scope="module")
def browser(app_url):
    b = cdp.Browser(port=9335, width=1500, height=900)
    b.goto(app_url, "Finding insiders from everyday activity", 120); time.sleep(2)
    b.click_text("New prediction"); assert b.wait("Rate a new, hypothetical worker"); time.sleep(3)
    type_into(b, "Worker name", "Browser worker"); b.eval("document.activeElement.blur()"); time.sleep(2.5)
    pick(b, "Websites visited", "wikileaks")
    assert b.wait("Visits to wikileaks.org"), "the website was not selected"
    press(b, "Add to this session's list")
    assert b.wait("Added Browser worker"), "the worker was not added"
    press(b, "Show on map")
    for _ in range(90):
        if b.eval("!!(window.__tmDebug && window.__tmDebug.cy.nodes('.hyp').length > 0)"): break
        time.sleep(1)
    else:
        pytest.fail("the hypothetical worker never appeared on the map")
    time.sleep(2)
    yield b
    b.close()


def test_worker_node_has_tag_legend_and_its_site_is_visible(browser):
    d = json.loads(browser.eval("""(() => { const cy = window.__tmDebug.cy, n = cy.nodes('.hyp')[0];
      return JSON.stringify({label: n.data('label'), nb: n.neighborhood('node').map((x) => x.data('label')), legend: window.__tmDebug.root.querySelector('.legend').innerText, tag: n.hasClass('labeled')}); })()"""))
    assert "hypothetical" in d["label"] and any("wikileaks.org" in x for x in d["nb"]) and "hypothetical worker" in d["legend"]


def test_hover_click_and_drag_the_worker_node_after_a_scroll(browser):
    browser.eval("window.__tmDebug.cy.center(window.__tmDebug.cy.nodes('.hyp')[0])"); time.sleep(0.3)   # pan only: put the worker mid-canvas so it is on screen after the scroll
    browser.eval("document.querySelector('[data-testid=stMain]').scrollTo(0, 0)"); time.sleep(0.5)
    browser.wheel(1470, 400, SCROLL_PX); time.sleep(0.8)          # real wheel over the page margin (the focused worker's long panel would swallow a wheel over the map)
    node = json.loads(browser.eval("""(() => { const d = window.__tmDebug, cy = d.cy, r = d.root.querySelector('.canvas').getBoundingClientRect(), n = cy.nodes('.hyp')[0];
      return JSON.stringify({id: n.id(), x: r.left + n.renderedPosition().x, y: r.top + n.renderedPosition().y, zoom: cy.zoom(), scroll: document.querySelector('[data-testid=stMain]').scrollTop, vis: r.top + n.renderedPosition().y > 90 && r.top + n.renderedPosition().y < innerHeight - 30}); })()"""))
    assert node["vis"], "worker node is outside the viewport after the scroll"
    assert node["scroll"] >= SCROLL_PX * 0.8
    assert "hypothetical" in hover(browser, node)
    browser.click(node["x"], node["y"]); time.sleep(0.5)
    assert "Browser worker" in browser.eval("window.__tmDebug.root.querySelector('.panel').innerText") and "HYPOTHETICAL WORKER" in browser.eval("window.__tmDebug.root.querySelector('.panel').innerText")
    before = browser.eval(_POS % node["id"])
    browser.drag(node["x"], node["y"], node["x"] + 60, node["y"] + 40); time.sleep(0.8)
    after = browser.eval(_POS % node["id"])
    assert abs((after[0] - before[0]) * node["zoom"] - 60) < 8 and abs((after[1] - before[1]) * node["zoom"] - 40) < 8
