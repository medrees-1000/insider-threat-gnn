"""Threat map component (Streamlit Components v2): the vendored Cytoscape.js plus app/threat_map.js, inlined as one ES module."""
from pathlib import Path

import streamlit as st

HERE = Path(__file__).parent
CYTOSCAPE = HERE / "vendor" / "cytoscape-3.35.0.min.js"


def _js():
    # Cytoscape is a UMD bundle: give it a CommonJS-style `module` so it exports instead of touching `this`
    lib = CYTOSCAPE.read_text()
    return ("var module = {exports: {}}; var exports = module.exports;\n(function(){\n" + lib + "\n}).call(globalThis);\nconst cytoscape = module.exports;\n"
            + (HERE / "threat_map.js").read_text())


_COMPONENT = st.components.v2.component("itgnn_threat_map", html="", css=(HERE / "threat_map.css").read_text(), js=_js())


def render(payload, theme, on_open, focus=None, key="threat_map"):
    # a changed `data` argument does not re-run the component's JavaScript, so anything that changes the payload (the evaluation flag, the
    # theme) must change the key and mount a fresh instance
    """Mount the map. `on_open` is called (before the rerun) when the user clicks "Open in Investigate" in the details panel."""
    key = f"{key}_{'eval' if payload['meta'].get('evaluation') else 'plain'}_{theme['page']}"
    return _COMPONENT(key=key, data={"payload": payload, "theme": theme, "focus": focus}, on_open_change=on_open)
