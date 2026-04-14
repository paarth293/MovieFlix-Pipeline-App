"""
MoodFlix — Emotion-Based Movie Recommender
Streamlit frontend — mirrors the React UI with the same cinematic palette
"""

import os, json, ast, sys
import streamlit as st
import pandas as pd

# ── Make sure the backend module is importable ────────────────────────────────
BACKEND = os.path.join(os.path.dirname(__file__), "app", "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

try:
    from recommender import recommend, get_metrics, get_metadata
    BACKEND_OK = True
except Exception as e:
    BACKEND_OK = False
    BACKEND_ERR = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodFlix | Find Your Film",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS & GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
BURGUNDY   = "#6B1D3A"
TERRACOTTA = "#C4663A"
GOLD       = "#D4AF37"
CREAM      = "#F7F2E8"
CREAM_DK   = "#EDE6D6"
PARCHMENT  = "#F0E8D6"
CHARCOAL   = "#1C1A1E"
SLATE      = "#4A454F"
MUTED      = "#8A8090"
SAGE       = "#7A9E7E"
SAD_BLUE   = "#3A5A8C"
ANGRY_RED  = "#8B2D1A"
RELAX_GRN  = "#3A6B3E"
NEUTRAL_PRP= "#5A4A6B"

MOOD_COLORS = {
    "Happy":   {"accent": TERRACOTTA, "bg": "#FFF3DC", "text": "#8B3A1A"},
    "Sad":     {"accent": SAD_BLUE,   "bg": "#EDF0F7", "text": "#2A3F6B"},
    "Angry":   {"accent": ANGRY_RED,  "bg": "#FBE9E7", "text": "#6B1D0E"},
    "Relaxed": {"accent": RELAX_GRN,  "bg": "#EBF3EC", "text": "#2A4F2E"},
    "Neutral": {"accent": NEUTRAL_PRP,"bg": "#F5F0F8", "text": "#3A2A4B"},
}

def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400;1,600&family=Lato:wght@300;400;700&family=DM+Mono:wght@400;500&display=swap');

    /* ── Root ── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {CREAM} !important;
        color: {CHARCOAL};
        font-family: 'Lato', sans-serif;
    }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background-color: {CHARCOAL} !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    [data-testid="stSidebar"] * {{ color: {CREAM} !important; }}
    [data-testid="stSidebarNav"] {{ display: none; }}

    /* ── Top bar hidden ── */
    header[data-testid="stHeader"] {{ background: {CHARCOAL}; }}
    #MainMenu, footer {{ visibility: hidden; }}

    /* ── Main content ── */
    .main .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: {CREAM_DK}; }}
    ::-webkit-scrollbar-thumb {{ background: {BURGUNDY}; border-radius: 3px; }}

    /* ── Hero section ── */
    .mf-hero {{
        background: {CHARCOAL};
        padding: 64px 48px 52px;
        position: relative;
        overflow: hidden;
        border-left: 5px solid;
        border-image: linear-gradient(to bottom, {GOLD}, {TERRACOTTA}, {BURGUNDY}) 1;
    }}
    .mf-hero-bg {{
        position: absolute; right: -10px; top: 50%;
        transform: translateY(-50%);
        font-family: 'Playfair Display', serif;
        font-size: 14rem; font-weight: 700;
        color: rgba(255,255,255,0.025);
        pointer-events: none; line-height: 1;
        letter-spacing: 0.05em;
    }}
    .mf-eyebrow {{
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem; letter-spacing: 0.14em;
        text-transform: uppercase; color: {TERRACOTTA};
        margin-bottom: 14px;
    }}
    .mf-hero-title {{
        font-family: 'Playfair Display', serif;
        font-size: 3.6rem; font-weight: 700;
        color: {CREAM}; line-height: 1.1; margin-bottom: 18px;
    }}
    .mf-hero-title em {{ color: {GOLD}; font-style: italic; }}
    .mf-hero-desc {{
        font-size: 1rem; color: rgba(247,242,232,0.68);
        line-height: 1.75; max-width: 520px; margin-bottom: 0;
    }}

    /* ── Stats bar ── */
    .mf-stats-bar {{
        background: {BURGUNDY};
        padding: 22px 48px;
        display: flex; gap: 48px; flex-wrap: wrap;
    }}
    .mf-stat {{ display: flex; flex-direction: column; align-items: center; gap: 3px; }}
    .mf-stat-v {{
        font-family: 'Playfair Display', serif;
        font-size: 2rem; font-weight: 700; color: {GOLD};
    }}
    .mf-stat-l {{
        font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: rgba(247,242,232,0.62);
    }}

    /* ── Section ── */
    .mf-section {{ padding: 52px 48px; }}
    .mf-section-dark {{ background: {CHARCOAL}; padding: 52px 48px; }}
    .mf-sec-ey {{
        font-family: 'DM Mono', monospace; font-size: 0.7rem;
        text-transform: uppercase; letter-spacing: 0.12em;
        color: {TERRACOTTA}; display: block; margin-bottom: 8px;
    }}
    .mf-sec-title {{
        font-family: 'Playfair Display', serif;
        font-size: 2rem; font-weight: 700; color: {CHARCOAL};
        margin-bottom: 32px;
    }}
    .mf-sec-title-lt {{ color: {CREAM}; }}

    /* ── Panel ── */
    .mf-panel {{
        background: {PARCHMENT};
        border: 1.5px solid {CREAM_DK};
        border-radius: 18px;
        padding: 26px 24px;
        margin-bottom: 20px;
    }}
    .mf-panel-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1rem; font-weight: 700; color: {CHARCOAL};
        display: flex; align-items: center; gap: 10px;
        margin-bottom: 18px;
    }}
    .mf-pnum {{
        font-family: 'DM Mono', monospace; font-size: 0.68rem;
        color: {TERRACOTTA}; background: rgba(196,102,58,0.1);
        border: 1px solid rgba(196,102,58,0.25);
        padding: 2px 7px; border-radius: 4px;
    }}

    /* ── Mood tile ── */
    .mood-tile {{
        display: flex; flex-direction: column;
        align-items: center; gap: 5px;
        padding: 18px 10px 14px;
        border-radius: 18px; border: 2px solid transparent;
        cursor: pointer; text-align: center;
        transition: all 0.2s ease;
    }}
    .mood-tile:hover {{ transform: translateY(-2px); }}
    .mood-tile.selected {{ transform: translateY(-3px); }}
    .mood-em {{ font-size: 2.2rem; line-height: 1; display: block; }}
    .mood-lb {{
        font-family: 'Playfair Display', serif;
        font-size: 0.92rem; font-weight: 700;
    }}
    .mood-sb {{ font-size: 0.68rem; color: {MUTED}; letter-spacing: 0.03em; }}

    /* ── Movie card ── */
    .mc-card {{
        background: var(--mc-bg, {CREAM});
        border: 1.5px solid rgba(107,29,58,0.12);
        border-radius: 18px; padding: 24px 22px 18px;
        position: relative; overflow: hidden;
        transition: all 0.2s ease;
        margin-bottom: 0;
    }}
    .mc-card::before {{
        content: ''; position: absolute;
        left: 0; top: 0; bottom: 0; width: 4px;
        border-radius: 4px 0 0 4px;
    }}
    .mc-rank {{
        position: absolute; top: 14px; right: 16px;
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem; color: {MUTED};
    }}
    .mc-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem; font-weight: 700; color: {CHARCOAL};
        margin-bottom: 9px; padding-right: 36px; line-height: 1.3;
    }}
    .mc-genres {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px; }}
    .gc {{
        padding: 2px 9px;
        border-radius: 16px; font-size: 0.68rem;
        font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;
    }}
    .mc-overview {{
        font-size: 0.84rem; color: {SLATE};
        line-height: 1.65; margin-bottom: 10px;
    }}
    .mc-footer {{
        display: flex; gap: 16px; margin-top: 12px;
        padding-top: 12px; border-top: 1px solid rgba(107,29,58,0.1);
        flex-wrap: wrap;
    }}
    .mc-meta {{ display: flex; flex-direction: column; gap: 2px; }}
    .mc-ml {{
        font-size: 0.62rem; text-transform: uppercase;
        letter-spacing: 0.08em; color: {MUTED}; font-weight: 700;
    }}
    .mc-mv {{
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem; color: {CHARCOAL}; font-weight: 500;
    }}

    /* ── Metrics ── */
    .metric-card {{
        background: {PARCHMENT};
        border: 1.5px solid {CREAM_DK};
        border-radius: 14px; padding: 22px 20px;
        text-align: center;
    }}
    .metric-val {{
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem; font-weight: 700; color: {BURGUNDY};
    }}
    .metric-lbl {{
        font-size: 0.76rem; text-transform: uppercase;
        letter-spacing: 0.09em; color: {MUTED}; font-weight: 700;
    }}

    /* ── Table ── */
    .mf-table {{
        width: 100%; border-collapse: collapse;
        font-size: 0.9rem; margin-top: 8px;
    }}
    .mf-table th {{
        padding: 10px 14px; text-align: left;
        font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.08em; color: {MUTED};
        border-bottom: 2px solid {CREAM_DK};
        background: {PARCHMENT};
    }}
    .mf-table td {{
        padding: 12px 14px; border-bottom: 1px solid {CREAM_DK};
        font-family: 'DM Mono', monospace; font-size: 0.86rem; color: {SLATE};
    }}
    .mf-table .best-row td {{ background: rgba(107,29,58,0.04); }}
    .mf-table .f1-cell {{ color: {BURGUNDY}; font-weight: 700; }}
    .best-badge {{
        display: inline-block; padding: 2px 8px;
        background: {GOLD}; color: {CHARCOAL};
        border-radius: 16px; font-size: 0.64rem;
        font-weight: 700; letter-spacing: 0.04em;
        margin-right: 6px; font-family: 'Lato', sans-serif;
    }}

    /* ── Bar chart ── */
    .bar-track {{
        height: 10px; background: {CREAM_DK};
        border-radius: 5px; overflow: hidden;
        flex: 1;
    }}
    .bar-fill {{ height: 100%; border-radius: 5px; }}
    .bar-row {{
        display: grid;
        grid-template-columns: 130px 1fr 58px;
        align-items: center; gap: 12px; margin-bottom: 14px;
    }}
    .bar-label {{
        font-family: 'DM Mono', monospace; font-size: 0.8rem;
        color: {CHARCOAL}; font-weight: 500;
    }}
    .bar-val {{
        font-family: 'DM Mono', monospace; font-size: 0.8rem;
        color: {SLATE}; text-align: right;
    }}

    /* ── Step card ── */
    .step-card {{
        padding: 22px 20px;
        background: {PARCHMENT};
        border: 1.5px solid {CREAM_DK};
        border-radius: 18px;
        height: 100%;
    }}
    .step-n {{
        font-family: 'Playfair Display', serif;
        font-size: 2rem; font-weight: 700;
        color: {BURGUNDY}; opacity: 0.25;
        line-height: 1; margin-bottom: 10px;
    }}
    .step-t {{
        font-family: 'Playfair Display', serif;
        font-size: 0.95rem; font-weight: 700;
        color: {CHARCOAL}; margin-bottom: 7px;
    }}
    .step-b {{ font-size: 0.84rem; color: {SLATE}; line-height: 1.6; }}

    /* ── Feature card ── */
    .feat-card {{
        padding: 24px 20px;
        background: rgba(255,255,255,0.04);
        border: 1.5px solid rgba(255,255,255,0.07);
        border-radius: 18px; height: 100%;
        transition: border-color 0.2s ease;
    }}
    .feat-ic {{ font-size: 1.6rem; margin-bottom: 12px; display: block; }}
    .feat-t {{
        font-family: 'Playfair Display', serif;
        font-size: 0.95rem; color: {GOLD}; margin-bottom: 8px;
    }}
    .feat-b {{ font-size: 0.8rem; color: rgba(247,242,232,0.55); line-height: 1.6; }}

    /* ── About ── */
    .tech-row {{
        display: flex; align-items: flex-start; gap: 10px;
        padding: 12px 14px; background: {CREAM};
        border-radius: 10px; border: 1.5px solid {CREAM_DK};
        margin-bottom: 10px;
    }}
    .tech-ic {{ font-size: 1.2rem; flex-shrink: 0; }}
    .tech-l {{
        font-size: 0.62rem; text-transform: uppercase;
        letter-spacing: 0.08em; color: {MUTED}; font-weight: 700; margin-bottom: 2px;
    }}
    .tech-n {{ font-size: 0.78rem; color: {CHARCOAL}; font-weight: 600; }}

    .run-step {{ display: flex; gap: 14px; margin-bottom: 16px; }}
    .rs-num {{
        width: 28px; height: 28px; flex-shrink: 0;
        background: {BURGUNDY}; color: {CREAM}; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Playfair Display', serif; font-size: 0.9rem; font-weight: 700;
        margin-top: 2px;
    }}
    .rs-t {{ font-size: 0.88rem; font-weight: 700; color: {CHARCOAL}; margin-bottom: 4px; }}
    .rs-code {{
        display: block; background: {CHARCOAL}; color: {GOLD};
        font-family: 'DM Mono', monospace; font-size: 0.78rem;
        padding: 7px 12px; border-radius: 5px; margin-bottom: 4px;
    }}
    .rs-note {{ font-size: 0.74rem; color: {MUTED}; }}

    .map-row {{
        display: flex; align-items: center; gap: 16px;
        padding: 10px 16px; background: {CREAM};
        border-radius: 10px; border-left: 4px solid;
        margin-bottom: 8px;
    }}
    .map-mood {{ font-size: 0.9rem; font-weight: 600; color: {CHARCOAL}; min-width: 110px; }}
    .map-arr {{ color: {MUTED}; }}
    .map-gen {{ font-family: 'Playfair Display', serif; font-size: 0.9rem; font-weight: 700; }}

    .pipe-card {{
        padding: 22px 20px; background: {PARCHMENT};
        border: 1.5px solid {CREAM_DK}; border-radius: 16px;
        margin-bottom: 14px;
    }}
    .pipe-head {{
        display: flex; align-items: center; gap: 10px; margin-bottom: 12px;
    }}
    .pipe-ic {{ font-size: 1.4rem; }}
    .pipe-t {{
        font-family: 'Playfair Display', serif;
        font-size: 0.95rem; font-weight: 700; color: {CHARCOAL};
    }}
    .pipe-ul {{ list-style: none; padding: 0; margin: 0; }}
    .pipe-li {{
        font-size: 0.8rem; color: {SLATE};
        padding-left: 14px; position: relative;
        line-height: 1.6; margin-bottom: 5px;
    }}

    /* ── Dist bar ── */
    .dist-row {{
        display: grid; grid-template-columns: 90px 1fr 58px 46px;
        align-items: center; gap: 12px; margin-bottom: 12px;
    }}
    .dist-n {{ font-family: 'Playfair Display', serif; font-size: 0.88rem; font-weight: 700; color: {CHARCOAL}; }}
    .dist-t {{ height: 10px; background: {CREAM_DK}; border-radius: 5px; overflow: hidden; }}
    .dist-f {{ height: 100%; border-radius: 5px; }}
    .dist-c {{ font-family: 'DM Mono', monospace; font-size: 0.76rem; color: {MUTED}; }}
    .dist-p {{ font-family: 'DM Mono', monospace; font-size: 0.8rem; font-weight: 600; color: {CHARCOAL}; text-align: right; }}

    /* ── Footer ── */
    .mf-footer {{
        background: {CHARCOAL}; padding: 22px 48px;
        display: flex; align-items: center;
        justify-content: space-between; gap: 12px; flex-wrap: wrap;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin-top: 0;
    }}
    .mf-ft-brand {{
        font-family: 'Playfair Display', serif;
        font-size: 1rem; color: {GOLD}; font-weight: 600;
    }}
    .mf-ft-txt {{
        font-family: 'DM Mono', monospace; font-size: 0.74rem; color: {SLATE};
    }}

    /* ── Sidebar nav ── */
    .sb-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem; font-weight: 700; color: {GOLD};
        padding: 20px 20px 8px; display: flex; align-items: center; gap: 8px;
    }}
    .sb-divider {{
        border: none; border-top: 1px solid rgba(255,255,255,0.08);
        margin: 8px 20px 16px;
    }}
    .sb-nav-item {{
        display: block; padding: 10px 20px; border-radius: 8px;
        font-size: 0.9rem; cursor: pointer;
        transition: background 0.2s ease;
        color: rgba(247,242,232,0.7) !important;
        margin: 2px 8px;
    }}
    .sb-nav-item:hover {{ background: rgba(255,255,255,0.06); color: {CREAM} !important; }}
    .sb-nav-item.active {{
        background: {BURGUNDY}; color: {CREAM} !important; font-weight: 700;
    }}
    .sb-section-lbl {{
        font-family: 'DM Mono', monospace; font-size: 0.62rem;
        text-transform: uppercase; letter-spacing: 0.12em;
        color: rgba(247,242,232,0.3); padding: 8px 20px 4px;
    }}

    /* ── Submit button ── */
    div[data-testid="stButton"] > button {{
        background: {CHARCOAL} !important;
        border: 2px solid {TERRACOTTA} !important;
        color: {CREAM} !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.05rem !important; font-style: italic !important;
        font-weight: 700 !important; border-radius: 10px !important;
        padding: 12px 28px !important;
        width: 100%; transition: all 0.2s ease !important;
    }}
    div[data-testid="stButton"] > button:hover {{
        background: {BURGUNDY} !important;
        border-color: {GOLD} !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(107,29,58,0.3) !important;
    }}

    /* ── Streamlit radio override ── */
    div[data-testid="stRadio"] > label {{
        font-family: 'Lato', sans-serif !important; font-size: 0.9rem !important;
    }}
    div[data-testid="stSelectbox"] label {{ color: {CHARCOAL} !important; }}

    /* ── CTA band ── */
    .mf-cta {{
        background: {TERRACOTTA}; padding: 60px 48px;
        text-align: center;
    }}
    .mf-cta-title {{
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem; font-weight: 700; color: {CREAM}; margin-bottom: 10px;
    }}
    .mf-cta-sub {{ font-size: 1rem; color: rgba(247,242,232,0.75); margin-bottom: 0; }}

    /* hide default padding top */
    .appview-container .main .block-container {{ padding-top: 0 !important; }}
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_stars(rating: float) -> str:
    filled = round((rating / 10) * 5)
    stars = "".join(
        f'<span style="color:{GOLD};font-size:0.95rem">★</span>' if i < filled
        else f'<span style="color:{CREAM_DK};font-size:0.95rem">★</span>'
        for i in range(5)
    )
    return f'{stars} <span style="font-family:\'DM Mono\',monospace;font-size:0.8rem;color:{SLATE};margin-left:4px">{rating}/10</span>'


def render_movie_card(movie: dict, idx: int) -> str:
    mc = MOOD_COLORS.get(movie.get("mood", "Neutral"), MOOD_COLORS["Neutral"])
    genres = movie.get("genres", [])
    genre_chips = "".join(
        f'<span class="gc" style="background:rgba({",".join(str(int(mc["accent"].lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.1);'
        f'border:1px solid rgba({",".join(str(int(mc["accent"].lstrip("#")[i:i+2],16)) for i in (0,2,4))},0.2);'
        f'color:{mc["accent"]}">{g}</span>'
        for g in genres
    )
    stars_html = render_stars(movie.get("rating", 0))
    overview = movie.get("overview", "No overview available.")

    return f"""
    <div class="mc-card" style="background:{mc['bg']};margin-bottom:16px">
        <div style="position:absolute;left:0;top:0;bottom:0;width:4px;background:{mc['accent']};border-radius:4px 0 0 4px"></div>
        <div class="mc-rank">#{idx+1}</div>
        <div class="mc-title">{movie.get('title','')}</div>
        <div class="mc-genres">{genre_chips}</div>
        <div class="mc-stars" style="margin-bottom:10px">{stars_html}</div>
        <p class="mc-overview">{overview}</p>
        <div class="mc-footer">
            <div class="mc-meta"><span class="mc-ml">Mood</span><span class="mc-mv">{movie.get('mood','')}</span></div>
            <div class="mc-meta"><span class="mc-ml">Rating</span><span class="mc-mv">{movie.get('rating',0)}/10</span></div>
            <div class="mc-meta"><span class="mc-ml">Popularity</span><span class="mc-mv">{movie.get('popularity',0)}</span></div>
            <div class="mc-meta"><span class="mc-ml">Cluster</span><span class="mc-mv">#{movie.get('cluster',0)}</span></div>
        </div>
    </div>"""


def bar_row_html(label, pct, color, val_str):
    return f"""
    <div class="bar-row">
        <span class="bar-label">{label}</span>
        <div class="bar-track"><div class="bar-fill" style="width:{pct:.2f}%;background:{color}"></div></div>
        <span class="bar-val">{val_str}</span>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "home"

with st.sidebar:
    st.markdown(f'<div class="sb-title">🎬 MoodFlix</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-lbl">Navigation</div>', unsafe_allow_html=True)

    pages = [
        ("🏠", "Home",        "home"),
        ("🎬", "Get Recs",    "rec"),
        ("📊", "ML Metrics",  "met"),
        ("ℹ️",  "About",       "abt"),
    ]

    for icon, label, pid in pages:
        active_cls = "active" if st.session_state.page == pid else ""
        if st.button(f"{icon}  {label}", key=f"nav_{pid}", use_container_width=True):
            st.session_state.page = pid
            st.rerun()

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-lbl">Status</div>', unsafe_allow_html=True)
    if BACKEND_OK:
        st.markdown(
            f'<div style="padding:8px 20px;font-size:0.78rem;color:{SAGE}">✅ ML models loaded</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="padding:8px 20px;font-size:0.78rem;color:#C4663A">⚠️ Models not loaded<br><small style="color:{MUTED}">Run train.py first</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="padding:8px 20px 20px;font-size:0.72rem;color:rgba(247,242,232,0.3);font-family:DM Mono,monospace">'
        f'SVM · K-Means · TMDb</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────

def page_home():
    # Hero
    st.markdown(f"""
    <div class="mf-hero">
        <div class="mf-hero-bg">CINEMA</div>
        <p class="mf-eyebrow">Machine Learning · TMDb Dataset</p>
        <h1 class="mf-hero-title">Your Mood,<br><em>Your Movie.</em></h1>
        <p class="mf-hero-desc">
            Tell us how you feel. Our SVM classifier and K-Means clustering
            pipeline will surface the films that match your emotional state —
            curated from 4,700+ titles using real ML.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats bar
    st.markdown(f"""
    <div style="background:{BURGUNDY};padding:22px 48px;display:flex;gap:60px;flex-wrap:wrap">
        <div class="mf-stat"><span class="mf-stat-v">92.88%</span><span class="mf-stat-l">SVM Accuracy</span></div>
        <div class="mf-stat"><span class="mf-stat-v">4,775</span><span class="mf-stat-l">Movies in dataset</span></div>
        <div class="mf-stat"><span class="mf-stat-v">11</span><span class="mf-stat-l">PCA Components</span></div>
        <div class="mf-stat"><span class="mf-stat-v">5</span><span class="mf-stat-l">K-Means Clusters</span></div>
        <div class="mf-stat"><span class="mf-stat-v">3</span><span class="mf-stat-l">ML Models compared</span></div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown(f"""
    <div class="mf-section">
        <div style="text-align:center;margin-bottom:36px">
            <span class="mf-sec-ey">The Pipeline</span>
            <div class="mf-sec-title" style="text-align:center">How MoodFlix Works</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("01", "Pick a Mood", "Choose from Happy, Sad, Angry, Relaxed, or Neutral to start."),
        ("02", "SVM Predicts", "Input is PCA-transformed and fed to the trained SVM classifier."),
        ("03", "Cluster Refines", "K-Means pinpoints the best-matching cluster of movies."),
        ("04", "Top Films Served", "Results sorted by rating, with genres and overview details."),
    ]
    for col, (n, t, b) in zip([c1,c2,c3,c4], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-n">{n}</div>
                <div class="step-t">{t}</div>
                <div class="step-b">{b}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ML Architecture (dark section)
    st.markdown(f"""
    <div class="mf-section-dark">
        <div style="text-align:center;margin-bottom:36px">
            <span class="mf-sec-ey">Under the Hood</span>
            <div class="mf-sec-title mf-sec-title-lt" style="text-align:center">ML Architecture</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    feats = [
        ("🧠", "SVM Classification", "RBF kernel achieves 92.88% accuracy mapping mood to genre category."),
        ("🔍", "K-Means Clustering", "5 clusters by genre and quality signals refine results beyond label filtering."),
        ("📐", "PCA Reduction", "11 components capture 90.9% variance across 20+ genre features."),
        ("🎬", "4,775 Movies", "TMDb 5000 cleaned with mood labels, one-hot genres and normalised scores."),
    ]
    for col, (ic, t, b) in zip([f1,f2,f3,f4], feats):
        with col:
            st.markdown(f"""
            <div class="feat-card">
                <span class="feat-ic">{ic}</span>
                <div class="feat-t">{t}</div>
                <div class="feat-b">{b}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # CTA
    st.markdown(f"""
    <div class="mf-cta">
        <div class="mf-cta-title">Ready to find tonight's film?</div>
        <p class="mf-cta-sub">Takes two seconds. No sign-up required.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    col_cta = st.columns([2,1,2])[1]
    with col_cta:
        if st.button("✨ Get Recommendations →", use_container_width=True):
            st.session_state.page = "rec"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RECOMMEND
# ─────────────────────────────────────────────────────────────────────────────

def page_rec():
    st.markdown(f"""
    <div class="rec-hero" style="background:{CHARCOAL};padding:52px 48px 44px;border-bottom:3px solid {BURGUNDY}">
        <p class="mf-eyebrow">Powered by SVM + K-Means</p>
        <h1 style="font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:700;color:{CREAM};margin-bottom:10px">
            How are you feeling?
        </h1>
        <p style="font-size:0.95rem;color:rgba(247,242,232,0.58)">
            Choose your current mood and let our ML pipeline find the perfect film.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Step 01: Mood ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="mf-panel">
        <div class="mf-panel-title"><span class="mf-pnum">01</span> Select Your Mood</div>
    </div>""", unsafe_allow_html=True)

    moods_data = [
        ("😄", "Happy",   "Comedy · Feel-good",    TERRACOTTA, "#FFF3DC"),
        ("😢", "Sad",     "Drama · Emotional",      SAD_BLUE,   "#EDF0F7"),
        ("😤", "Angry",   "Action · Adrenaline",   ANGRY_RED,  "#FBE9E7"),
        ("😌", "Relaxed", "Romance · Heartwarming",RELAX_GRN,  "#EBF3EC"),
        ("😐", "Neutral", "Mixed · Any genre",      NEUTRAL_PRP,"#F5F0F8"),
    ]

    if "sel_mood" not in st.session_state:
        st.session_state.sel_mood = "Happy"

    cols = st.columns(5)
    for col, (em, label, sub, color, bg) in zip(cols, moods_data):
        with col:
            selected = st.session_state.sel_mood == label
            border_style = f"2px solid {color}" if selected else f"2px solid transparent"
            transform = "translateY(-3px)" if selected else "none"
            shadow = f"0 6px 20px rgba(0,0,0,0.12)" if selected else "none"
            st.markdown(f"""
            <div class="mood-tile {'selected' if selected else ''}"
                 style="background:{bg};border:{border_style};transform:{transform};box-shadow:{shadow}">
                <span class="mood-em">{em}</span>
                <span class="mood-lb" style="color:{color}">{label}</span>
                <span class="mood-sb">{sub}</span>
                {'<span style="position:absolute;top:10px;right:12px;width:20px;height:20px;background:'+color+';color:#fff;border-radius:50%;font-size:0.68rem;display:flex;align-items:center;justify-content:center;font-weight:700">✓</span>' if selected else ''}
            </div>""", unsafe_allow_html=True)
            if st.button(label, key=f"mood_{label}", use_container_width=True):
                st.session_state.sel_mood = label
                st.rerun()

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Step 02 & 03 ──────────────────────────────────────────────────────────
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown(f"""
        <div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:24px 22px;margin-bottom:0">
            <div class="mf-panel-title"><span class="mf-pnum">02</span> ML Model</div>
        </div>""", unsafe_allow_html=True)

        model_map = {
            "SVM  —  Primary · Best accuracy": "svm",
            "KNN  —  K-Nearest Neighbours": "knn",
            "Logistic Regression  —  Linear baseline": "lr",
        }
        sel_model_label = st.radio(
            "model_pick", list(model_map.keys()),
            label_visibility="collapsed",
        )
        sel_model = model_map[sel_model_label]

    with right_col:
        st.markdown(f"""
        <div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:24px 22px;margin-bottom:0">
            <div class="mf-panel-title"><span class="mf-pnum">03</span> Number of Results</div>
        </div>""", unsafe_allow_html=True)

        top_n = st.select_slider(
            "top_n_slider", options=[3, 5, 8, 10],
            value=5, label_visibility="collapsed",
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Submit ─────────────────────────────────────────────────────────────────
    if st.button(
        f'✨ Get Recommendations for "{st.session_state.sel_mood}" →',
        use_container_width=True,
    ):
        if not BACKEND_OK:
            st.error(f"⚠️ Backend not available. Run `python train.py` first.\n\n{BACKEND_ERR}")
        else:
            with st.spinner("Analysing your mood with SVM…"):
                movies, label = recommend(st.session_state.sel_mood, sel_model, top_n)

            if not movies:
                st.error(f"No movies found for mood: {st.session_state.sel_mood}")
            else:
                mc_color = MOOD_COLORS.get(st.session_state.sel_mood, MOOD_COLORS["Neutral"])
                st.markdown(f"""
                <div style="display:flex;align-items:flex-start;justify-content:space-between;
                    flex-wrap:wrap;gap:12px;margin:24px 0 20px;
                    padding-bottom:14px;border-bottom:1.5px solid {CREAM_DK}">
                    <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;color:{CHARCOAL}">
                        Top {len(movies)} films for <em style="color:{TERRACOTTA}">{st.session_state.sel_mood}</em>
                    </div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
                        <span style="padding:3px 11px;background:{CHARCOAL};color:{GOLD};
                            border-radius:20px;font-family:'DM Mono',monospace;font-size:0.72rem">
                            Model: {sel_model.upper()}</span>
                        <span style="padding:3px 11px;background:{CHARCOAL};color:{GOLD};
                            border-radius:20px;font-family:'DM Mono',monospace;font-size:0.72rem">
                            Predicted: {label}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                # 2-column movie grid
                g1, g2 = st.columns(2)
                for i, movie in enumerate(movies):
                    col = g1 if i % 2 == 0 else g2
                    with col:
                        st.markdown(render_movie_card(movie, i), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: METRICS
# ─────────────────────────────────────────────────────────────────────────────

def page_metrics():
    st.markdown(f"""
    <div style="background:{CHARCOAL};padding:52px 48px 44px;border-bottom:3px solid {TERRACOTTA}">
        <p class="mf-eyebrow" style="color:{TERRACOTTA}">Evaluation · Results · Pipeline</p>
        <h1 style="font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:700;color:{CREAM};margin-bottom:10px">
            ML Model Performance
        </h1>
        <p style="font-size:0.95rem;color:rgba(247,242,232,0.58)">
            Detailed evaluation of SVM, KNN and Logistic Regression trained on the TMDb dataset.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # Load metrics
    if BACKEND_OK:
        metrics = get_metrics()
    else:
        metrics = {
            "SVM": {"accuracy":0.9288,"precision":0.9287,"recall":0.9288,"f1":0.9244},
            "KNN": {"accuracy":0.9141,"precision":0.9110,"recall":0.9141,"f1":0.9088},
            "Logistic Regression": {"accuracy":0.9288,"precision":0.9311,"recall":0.9288,"f1":0.9241},
        }

    # Summary metric cards
    best = max(metrics, key=lambda m: metrics[m]["f1"])
    mc1,mc2,mc3,mc4 = st.columns(4)
    summary = [
        ("Best Model", best),
        ("SVM Accuracy", f"{metrics['SVM']['accuracy']*100:.2f}%"),
        ("SVM F1 Score", f"{metrics['SVM']['f1']*100:.2f}%"),
        ("Dataset size", "4,775"),
    ]
    for col, (lbl, val) in zip([mc1,mc2,mc3,mc4], summary):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">{lbl}</div>
                <div class="metric-val">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # Bar charts
    bl, br = st.columns(2)
    with bl:
        st.markdown(f"""
        <div class="chart-card" style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:26px 24px">
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{CHARCOAL};margin-bottom:4px">Accuracy by Model</div>
            <div style="font-size:0.78rem;color:{MUTED};margin-bottom:20px">All values as percentage (%)</div>
            {bar_row_html('SVM ⭐', metrics['SVM']['accuracy']*100, BURGUNDY, f"{metrics['SVM']['accuracy']*100:.2f}%")}
            {bar_row_html('Log. Reg.', metrics['Logistic Regression']['accuracy']*100, SAGE, f"{metrics['Logistic Regression']['accuracy']*100:.2f}%")}
            {bar_row_html('KNN', metrics['KNN']['accuracy']*100, TERRACOTTA, f"{metrics['KNN']['accuracy']*100:.2f}%")}
        </div>""", unsafe_allow_html=True)

    with br:
        st.markdown(f"""
        <div class="chart-card" style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:26px 24px">
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{CHARCOAL};margin-bottom:4px">F1 Score by Model</div>
            <div style="font-size:0.78rem;color:{MUTED};margin-bottom:20px">Weighted F1 as percentage (%)</div>
            {bar_row_html('SVM ⭐', metrics['SVM']['f1']*100, BURGUNDY, f"{metrics['SVM']['f1']*100:.2f}%")}
            {bar_row_html('Log. Reg.', metrics['Logistic Regression']['f1']*100, SAGE, f"{metrics['Logistic Regression']['f1']*100:.2f}%")}
            {bar_row_html('KNN', metrics['KNN']['f1']*100, TERRACOTTA, f"{metrics['KNN']['f1']*100:.2f}%")}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Scorecard table
    def _srows(met, b):
        out = ""
        for m in met:
            rc = "best-row" if m == b else ""
            badge = '<span class="best-badge">&#9733; Best</span> ' if m == b else ""
            fc = "f1-cell" if m == b else ""
            out += ('<tr class="' + rc + '"><td><div class="mname">' + badge + m + '</div></td>'
                    + '<td>' + f"{met[m]['accuracy']*100:.2f}" + '%</td>'
                    + '<td>' + f"{met[m]['precision']*100:.2f}" + '%</td>'
                    + '<td>' + f"{met[m]['recall']*100:.2f}" + '%</td>'
                    + '<td class="' + fc + '">' + f"{met[m]['f1']*100:.2f}" + '%</td></tr>')
        return out

    sc_head = (
        f'<div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:26px 24px">'
        f'<div style="font-family:Playfair Display,serif;font-size:1.1rem;font-weight:700;color:{CHARCOAL};margin-bottom:4px">Model Scorecard</div>'
        f'<div style="font-size:0.78rem;color:{MUTED};margin-bottom:20px">&#9733; highlights best-performing model</div>'
        '<div style="overflow-x:auto"><table class="mf-table">'
        '<thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr></thead>'
    )
    sc_html = sc_head + '<tbody>' + _srows(metrics, best) + '</tbody></table></div></div>'
    st.markdown(sc_html, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Mood distribution
    st.markdown(f"""
    <div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:26px 24px">
        <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{CHARCOAL};margin-bottom:4px">Mood Label Distribution</div>
        <div style="font-size:0.78rem;color:{MUTED};margin-bottom:20px">From 4,775 cleaned movies after preprocessing</div>
        {''.join(f"""<div class="dist-row">
            <span class="dist-n">{n}</span>
            <div class="dist-t"><div class="dist-f" style="width:{p}%;background:{c}"></div></div>
            <span class="dist-c">{cnt:,}</span>
            <span class="dist-p">{p}%</span>
        </div>""" for n,cnt,p,c in [
            ("Sad",1602,33.6,SAD_BLUE),("Happy",1299,27.2,TERRACOTTA),
            ("Angry",1034,21.7,ANGRY_RED),("Neutral",718,15.0,NEUTRAL_PRP),("Relaxed",122,2.6,RELAX_GRN)
        ])}
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Pipeline walkthrough
    st.markdown(f"""
    <div style="font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:700;
        color:{CHARCOAL};margin-bottom:18px">ML Pipeline Walkthrough</div>""",
        unsafe_allow_html=True,
    )
    pl1, pl2 = st.columns(2)
    pipe_steps = [
        ("🧹","Data Preprocessing",
         ["Parsed JSON genre column → genre name list","Dropped missing values & genre-less rows",
          "One-hot encoded 20 genres","StandardScaler on popularity & vote_average","Mapped genres → mood labels"]),
        ("📐","PCA Feature Engineering",
         ["22 features: 20 genre flags + 2 scaled numerics","11 principal components explain 90.9% variance",
          "Reduces noise before classification","Speeds up SVM inference"]),
        ("🤖","SVM Classification (Primary)",
         ["RBF kernel, C=5.0, gamma=scale","80/20 train-test split, stratified",
          "92.88% accuracy on held-out test set","Probability estimates enabled"]),
        ("🔍","K-Means Clustering",
         ["Elbow method tested K=2…12","K=5 chosen as optimal inflection",
          "Cluster assignment refines mood-filtered results","Query vector projected into cluster space"]),
    ]
    for col, (ic, t, items) in zip([pl1,pl2,pl1,pl2], pipe_steps):
        with col:
            li_html = "".join(
                f'<div class="pipe-li" style="font-size:0.8rem;color:{SLATE};padding-left:14px;'
                f'position:relative;line-height:1.6;margin-bottom:5px">'
                f'<span style="position:absolute;left:0;color:{TERRACOTTA};font-size:0.72rem;font-weight:700">→</span>'
                f'{item}</div>'
                for item in items
            )
            st.markdown(f"""
            <div class="pipe-card" style="margin-bottom:16px">
                <div class="pipe-head">
                    <span class="pipe-ic">{ic}</span>
                    <span class="pipe-t">{t}</span>
                </div>
                {li_html}
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────

def page_about():
    st.markdown(f"""
    <div style="background:{CHARCOAL};padding:52px 48px 44px;border-bottom:3px solid {GOLD}">
        <p class="mf-eyebrow" style="color:{GOLD}">Academic ML Project</p>
        <h1 style="font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:700;color:{CREAM};margin-bottom:10px">
            About MoodFlix
        </h1>
        <p style="font-size:0.95rem;color:rgba(247,242,232,0.58)">
            End-to-end emotion-based recommendation system built with scikit-learn, Flask &amp; Streamlit.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # Overview
    st.markdown(f"""
    <div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:28px 24px;margin-bottom:20px">
        <div style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;
            color:{CHARCOAL};margin-bottom:12px;padding-bottom:10px;border-bottom:1.5px solid {CREAM_DK}">
            Project Overview
        </div>
        <p style="font-size:0.9rem;color:{SLATE};line-height:1.75;margin-bottom:12px">
            MoodFlix demonstrates a complete ML pipeline from raw data to a deployed recommendation interface.
            The user's mood is converted into a feature vector, classified by an SVM model trained on TMDb genre
            data, then refined by K-Means clustering to surface the most relevant films.
        </p>
        <p style="font-size:0.9rem;color:{SLATE};line-height:1.75">
            Covers every core academic ML component: data preprocessing, feature engineering (PCA),
            multi-class classification (SVM, KNN, Logistic Regression), unsupervised clustering,
            and a full evaluation suite with accuracy, precision, recall, F1, and confusion matrices.
        </p>
    </div>""", unsafe_allow_html=True)

    # Tech stack
    st.markdown(f"""
    <div style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;
        color:{CHARCOAL};margin-bottom:16px;padding-bottom:10px;border-bottom:1.5px solid {CREAM_DK}">
        Technology Stack
    </div>""", unsafe_allow_html=True)

    tech = [
        ("🎬","Dataset","TMDb 5000 Movies"),("🧹","Preprocessing","Pandas · StandardScaler"),
        ("📐","Feature Eng.","PCA (scikit-learn)"),("🤖","Classification","SVM · KNN · LogReg"),
        ("🔍","Clustering","K-Means (K=5)"),("💾","Serialisation","joblib (.pkl)"),
        ("⚙️","Backend API","Flask · Flask-CORS"),("🖥️","Frontend","Streamlit · Recharts"),
    ]
    t1,t2,t3,t4 = st.columns(4)
    for i, (ic, layer, name) in enumerate(tech):
        col = [t1,t2,t3,t4][i%4]
        with col:
            st.markdown(f"""
            <div class="tech-row" style="margin-bottom:10px">
                <span class="tech-ic">{ic}</span>
                <div>
                    <div class="tech-l">{layer}</div>
                    <div class="tech-n">{name}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Run instructions
    st.markdown(f"""
    <div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:28px 24px;margin-bottom:20px">
        <div style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;
            color:{CHARCOAL};margin-bottom:16px;padding-bottom:10px;border-bottom:1.5px solid {CREAM_DK}">
            How to Run
        </div>
        {''.join(f"""<div class="run-step">
            <div class="rs-num">{n}</div>
            <div>
                <div class="rs-t">{t}</div>
                <code class="rs-code">{cmd}</code>
                <div class="rs-note">{note}</div>
            </div>
        </div>""" for n,t,cmd,note in [
            ("1","Install dependencies","pip install -r requirements.txt","Python 3.8+"),
            ("2","Train the ML pipeline","python train.py","Saves models/ and visualizations/"),
            ("3","Run this Streamlit app","streamlit run streamlit_app.py","Opens at localhost:8501"),
            ("4","(Optional) Flask API","cd app/backend &amp;&amp; python app.py","Runs at localhost:5000"),
        ])}
    </div>""", unsafe_allow_html=True)

    # Mood mapping
    st.markdown(f"""
    <div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:28px 24px">
        <div style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;
            color:{CHARCOAL};margin-bottom:16px;padding-bottom:10px;border-bottom:1.5px solid {CREAM_DK}">
            Mood → Genre Mapping
        </div>
        {''.join(f"""<div class="map-row" style="border-left-color:{c};margin-bottom:8px">
            <span class="map-mood">{em} {mood}</span>
            <span class="map-arr">→</span>
            <span class="map-gen" style="color:{c}">{genre}</span>
        </div>""" for em,mood,genre,c in [
            ("😄","Happy","Comedy",TERRACOTTA),("😢","Sad","Drama",SAD_BLUE),
            ("😤","Angry","Action",ANGRY_RED),("😌","Relaxed","Romance",RELAX_GRN),
            ("😐","Neutral","Mixed",NEUTRAL_PRP),
        ])}
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Academic checklist
    st.markdown(f"""
    <div style="background:{CHARCOAL};border-radius:18px;padding:28px 24px">
        <div style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;
            color:{GOLD};margin-bottom:16px">Academic Components Demonstrated</div>
        {''.join(f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
            <span style="color:{SAGE};font-size:1rem">✅</span>
            <span style="font-size:0.88rem;color:rgba(247,242,232,0.8)">{item}</span>
        </div>""" for item in [
            "Data preprocessing — missing values, JSON parsing, encoding, normalisation",
            "Feature engineering — PCA with variance analysis (11 components, 90.9%)",
            "Classification — SVM (primary 92.88%), KNN, Logistic Regression",
            "Model evaluation — Accuracy, Precision, Recall, F1, Confusion Matrix",
            "Unsupervised clustering — K-Means with Elbow Method (K=5)",
            "Recommendation logic — classification + clustering pipeline",
            "Visualisations — genre distribution, PCA variance, confusion matrices, elbow plot",
            "Interactive UI — Streamlit frontend with cinematic design",
        ])}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

def render_footer():
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="mf-footer">
        <span class="mf-ft-brand">🎬 MoodFlix</span>
        <span class="mf-ft-txt">Powered by SVM · K-Means · TMDb Dataset</span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

p = st.session_state.page
if   p == "home": page_home()
elif p == "rec":  page_rec()
elif p == "met":  page_metrics()
elif p == "abt":  page_about()
else:             page_home()

render_footer()
