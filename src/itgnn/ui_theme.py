"""One palette for the whole UI (dark default, validated light variant). Risk colours are reserved for risk; one accent;
muted neutrals; categorical hues only for named-site categories. Contrast is enforced by tests/test_ui_theme.py and the
categorical/ordinal sets were run through the data-viz palette validator (see docs/PROJECT_LOG.md)."""

PALETTE = {
    "dark": {
        "page": "#0e1117", "panel": "#161c27", "canvas": "#0b1017", "border": "#263042",
        "text": "#e6edf3", "text2": "#a8b3c1", "muted": "#7d8a9b", "accent": "#4aa3ff",
        "risk": ["#f9b36a", "#f28a45", "#e96a36", "#d94a2d"],              # top 2%, 1%, 0.5%, 0.25% of weeks
        "site": "#8196b0", "pc": "#a39e93", "contact": "#6f7d90",          # website diamonds, PC squares, email contacts
        "cat": {"leak": "#8e4088", "download": "#19aa8e", "job": "#a07cdb", "cloud": "#2f751d", "competitor": "#8196b0"},
        "ring": "#ffffff", "edge": {"pc": "#7d8a9b", "dom": "#5c7794", "mail": "#4aa3ff"}, "positive": "#4aa3ff", "negative": "#6f7d90",
    },
    "light": {
        "page": "#fbfcfd", "panel": "#eef1f5", "canvas": "#f4f6f9", "border": "#d5dbe3",
        "text": "#101828", "text2": "#475467", "muted": "#5f6b7d", "accent": "#1a6fd6",
        "risk": ["#d98a3a", "#cc6a24", "#bf4a18", "#a8320f"],
        "site": "#5d7189", "pc": "#7a756a", "contact": "#7b8899",
        "cat": {"leak": "#81347c", "download": "#05a388", "job": "#9a76d4", "cloud": "#23690d", "competitor": "#5d7189"},
        "ring": "#101828", "edge": {"pc": "#667085", "dom": "#7a90a8", "mail": "#1a6fd6"}, "positive": "#1a6fd6", "negative": "#8794a6",
    },
}
CATEGORY_NOTE_SHORT = {"competitor": "competitor (neutral, dashed outline)"}


def _lin(c):
    c /= 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(hex_):
    h = hex_.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast(a, b):
    """WCAG contrast ratio of two hex colours."""
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def theme_for(mode):
    return PALETTE["light" if mode == "light" else "dark"]
