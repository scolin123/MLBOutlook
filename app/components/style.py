"""Look and feel: a clean, Savant-inspired scouting style. No third-party assets or logos."""

from __future__ import annotations

import streamlit as st

NAVY = "#0A2240"
RED = "#D22D49"
BLUE = "#2F6DB5"
GREY = "#8A94A6"
LIGHT = "#F4F6FA"
FORECAST = "#E8862A"

CSS = f"""
<style>
  .block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
  h1, h2, h3 {{ color: {NAVY}; letter-spacing: -0.01em; }}
  .diq-banner {{
    background: {LIGHT}; border-left: 4px solid {FORECAST}; padding: 0.55rem 0.9rem;
    border-radius: 4px; font-size: 0.9rem; color: #333; margin-bottom: 1rem;
  }}
  .diq-card {{
    border: 1px solid #E1E5EC; border-radius: 6px; padding: 0.8rem 1rem; background: white;
  }}
  .diq-card .label {{ font-size: 0.75rem; text-transform: uppercase; color: {GREY};
    letter-spacing: 0.06em; }}
  .diq-card .value {{ font-size: 1.6rem; font-weight: 700; color: {NAVY}; }}
  .diq-card .sub {{ font-size: 0.8rem; color: #555; }}
</style>
"""


def percentile_color(pct: float) -> str:
    """Savant-style diverging scale: blue (low) through grey to red (high)."""
    pct = max(0.0, min(100.0, float(pct))) / 100.0
    lo, mid, hi = (47, 109, 181), (190, 196, 206), (210, 45, 73)
    a, b, t = (lo, mid, pct * 2) if pct < 0.5 else (mid, hi, (pct - 0.5) * 2)
    r, g, bl = (round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return f"#{r:02X}{g:02X}{bl:02X}"


def setup(title: str, banner: str) -> None:
    """Page chrome plus the banner that says what season the page is about.

    `banner` has no default on purpose. The app now carries both kinds of page -- 2027
    forecasts with no outcome, and the scored folds with one -- and a page that forgot to say
    which it was would inherit the wrong claim silently. Making it an argument means the
    mislabelling cannot happen by omission.
    """
    st.set_page_config(page_title=f"DiamondIQ - {title}", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="diq-banner">{banner}</div>', unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="diq-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def woba(x: float) -> str:
    """Baseball style: .345, not 0.345."""
    return "-" if x != x else f"{x:.3f}".replace("0.", ".", 1)


def signed(x: float) -> str:
    return "-" if x != x else f"{x:+.3f}".replace("0.", ".", 1)


def pct(x: float) -> str:
    return "-" if x != x else f"{100 * x:.0f}%"
