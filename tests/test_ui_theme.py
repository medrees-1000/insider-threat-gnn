from itgnn.ui_theme import PALETTE, contrast


def test_text_and_accent_contrast():
    for mode, p in PALETTE.items():
        assert contrast(p["text"], p["page"]) >= 7, mode
        assert contrast(p["text2"], p["panel"]) >= 4.5 and contrast(p["muted"], p["panel"]) >= 4.5, mode
        assert contrast(p["accent"], p["canvas"]) >= 4.5, mode


def test_graph_marks_are_visible_on_the_canvas():
    for mode, p in PALETTE.items():
        assert all(contrast(p[k], p["canvas"]) >= 3.0 for k in ("site", "pc", "contact")), mode
        assert all(contrast(v, p["canvas"]) >= 3.0 for v in p["edge"].values()), mode
        assert all(contrast(c, p["canvas"]) >= 2.5 for c in p["risk"]), mode           # ordinal ramp rule (lightest step >= 2:1, kept at 2.5)
        assert all(contrast(c, p["canvas"]) >= 2.9 for c in p["cat"].values()), mode   # below 3:1 only with always-visible labels (validator "relief")
        assert contrast(p["ring"], p["canvas"]) >= 7, mode


def test_risk_scale_gets_darker_with_risk_and_is_reserved():
    for p in PALETTE.values():
        from itgnn.ui_theme import luminance
        lum = [luminance(c) for c in p["risk"]]
        assert lum == sorted(lum, reverse=True)          # amber (low) -> red (high): monotone darker
        assert not set(p["risk"]) & (set(p["cat"].values()) | {p["accent"], p["site"], p["pc"], p["contact"]})  # risk colours are not reused
