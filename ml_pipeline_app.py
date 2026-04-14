"""
MoodFlix ✦ ML Pipeline Lab
A cinematic fusion of emotion-based movie recommendation
and a full 9-step interactive ML pipeline dashboard.
"""

import os, json, sys, io, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

# ── Optional ML imports ────────────────────────────────────────────────────
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                  confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
    from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
    from sklearn.svm import SVC, SVR
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans, OPTICS, DBSCAN
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.pipeline import Pipeline
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ── MoodFlix backend ────────────────────────────────────────────────────────
BACKEND = os.path.join(os.path.dirname(__file__), "movie_recommender", "app", "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
try:
    from recommender import recommend, get_metrics, get_metadata
    MOODFLIX_OK = True
except Exception as e:
    MOODFLIX_OK = False
    MOODFLIX_ERR = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MoodFlix ✦ ML Lab",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
BURGUNDY    = "#6B1D3A"
TERRACOTTA  = "#C4663A"
GOLD        = "#D4AF37"
CREAM       = "#F7F2E8"
CREAM_DK    = "#EDE6D6"
PARCHMENT   = "#F0E8D6"
CHARCOAL    = "#1C1A1E"
SLATE       = "#4A454F"
MUTED       = "#8A8090"
SAGE        = "#7A9E7E"
SAD_BLUE    = "#3A5A8C"
ANGRY_RED   = "#8B2D1A"
RELAX_GRN   = "#3A6B3E"
NEUTRAL_PRP = "#5A4A6B"
TEAL        = "#2A7F7F"
AMBER       = "#E08B20"

MOOD_COLORS = {
    "Happy":   {"accent": TERRACOTTA, "bg": "#FFF3DC", "text": "#8B3A1A"},
    "Sad":     {"accent": SAD_BLUE,   "bg": "#EDF0F7", "text": "#2A3F6B"},
    "Angry":   {"accent": ANGRY_RED,  "bg": "#FBE9E7", "text": "#6B1D0E"},
    "Relaxed": {"accent": RELAX_GRN,  "bg": "#EBF3EC", "text": "#2A4F2E"},
    "Neutral": {"accent": NEUTRAL_PRP,"bg": "#F5F0F8", "text": "#3A2A4B"},
}

STEP_COLORS = [BURGUNDY, TERRACOTTA, GOLD, TEAL, SAGE, SAD_BLUE, NEUTRAL_PRP, ANGRY_RED, AMBER]

PIPELINE_STEPS = [
    ("🎯", "Problem Type"),
    ("📂", "Data Input"),
    ("🔍", "EDA"),
    ("🔧", "Engineering"),
    ("✂️", "Features"),
    ("✂️", "Data Split"),
    ("🤖", "Model"),
    ("🏋️", "Training"),
    ("📈", "Metrics"),
    ("⚙️", "Tuning"),
]

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400;1,600&family=Lato:wght@300;400;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {CREAM} !important;
        color: {CHARCOAL};
        font-family: 'Lato', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: {CHARCOAL} !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    [data-testid="stSidebar"] * {{ color: {CREAM} !important; }}
    [data-testid="stSidebarNav"] {{ display: none; }}
    header[data-testid="stHeader"] {{ background: {CHARCOAL}; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .main .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
    }}
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {CREAM_DK}; }}
    ::-webkit-scrollbar-thumb {{ background: {BURGUNDY}; border-radius: 3px; }}

    /* Hero */
    .mf-hero {{
        background: {CHARCOAL};
        padding: 64px 48px 52px;
        position: relative; overflow: hidden;
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
        line-height: 1.75; max-width: 520px;
    }}

    /* Stats bar */
    .mf-stat {{ display: flex; flex-direction: column; align-items: center; gap: 3px; }}
    .mf-stat-v {{
        font-family: 'Playfair Display', serif;
        font-size: 2rem; font-weight: 700; color: {GOLD};
    }}
    .mf-stat-l {{
        font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: rgba(247,242,232,0.62);
    }}

    /* Sections */
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

    /* Panel */
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

    /* Mood tiles */
    .mood-tile {{
        display: flex; flex-direction: column;
        align-items: center; gap: 5px;
        padding: 18px 10px 14px;
        border-radius: 18px; border: 2px solid transparent;
        cursor: pointer; text-align: center;
        transition: all 0.2s ease;
    }}
    .mood-tile:hover {{ transform: translateY(-2px); }}
    .mood-em {{ font-size: 2.2rem; line-height: 1; display: block; }}
    .mood-lb {{
        font-family: 'Playfair Display', serif;
        font-size: 0.92rem; font-weight: 700;
    }}
    .mood-sb {{ font-size: 0.68rem; color: {MUTED}; }}

    /* Movie card */
    .mc-card {{
        background: var(--mc-bg, {CREAM});
        border: 1.5px solid rgba(107,29,58,0.12);
        border-radius: 18px; padding: 24px 22px 18px;
        position: relative; overflow: hidden;
        margin-bottom: 0;
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
        padding: 2px 9px; border-radius: 16px; font-size: 0.68rem;
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

    /* Metrics */
    .metric-card {{
        background: {PARCHMENT}; border: 1.5px solid {CREAM_DK};
        border-radius: 14px; padding: 22px 20px; text-align: center;
    }}
    .metric-val {{
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem; font-weight: 700; color: {BURGUNDY};
    }}
    .metric-lbl {{
        font-size: 0.76rem; text-transform: uppercase;
        letter-spacing: 0.09em; color: {MUTED}; font-weight: 700;
    }}

    /* Bar chart */
    .bar-track {{ height: 10px; background: {CREAM_DK}; border-radius: 5px; overflow: hidden; flex: 1; }}
    .bar-fill {{ height: 100%; border-radius: 5px; }}
    .bar-row {{
        display: grid; grid-template-columns: 130px 1fr 58px;
        align-items: center; gap: 12px; margin-bottom: 14px;
    }}
    .bar-label {{ font-family: 'DM Mono', monospace; font-size: 0.8rem; color: {CHARCOAL}; font-weight: 500; }}
    .bar-val {{ font-family: 'DM Mono', monospace; font-size: 0.8rem; color: {SLATE}; text-align: right; }}

    /* Step card */
    .step-card {{
        padding: 22px 20px; background: {PARCHMENT};
        border: 1.5px solid {CREAM_DK}; border-radius: 18px; height: 100%;
    }}
    .step-n {{
        font-family: 'Playfair Display', serif;
        font-size: 2rem; font-weight: 700; color: {BURGUNDY};
        opacity: 0.25; line-height: 1; margin-bottom: 10px;
    }}
    .step-t {{
        font-family: 'Playfair Display', serif;
        font-size: 0.95rem; font-weight: 700; color: {CHARCOAL}; margin-bottom: 7px;
    }}
    .step-b {{ font-size: 0.84rem; color: {SLATE}; line-height: 1.6; }}

    /* Feature card (dark) */
    .feat-card {{
        padding: 24px 20px; background: rgba(255,255,255,0.04);
        border: 1.5px solid rgba(255,255,255,0.07);
        border-radius: 18px; height: 100%;
    }}
    .feat-ic {{ font-size: 1.6rem; margin-bottom: 12px; display: block; }}
    .feat-t {{ font-family: 'Playfair Display', serif; font-size: 0.95rem; color: {GOLD}; margin-bottom: 8px; }}
    .feat-b {{ font-size: 0.8rem; color: rgba(247,242,232,0.55); line-height: 1.6; }}

    /* Pipeline stepper */
    .pipeline-stepper {{
        background: {CHARCOAL};
        padding: 28px 40px 0;
        border-bottom: 2px solid rgba(255,255,255,0.06);
    }}
    .pipeline-header {{
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 24px;
    }}
    .pipeline-title {{
        font-family: 'Playfair Display', serif;
        font-size: 2rem; font-weight: 700; color: {CREAM};
    }}
    .pipeline-subtitle {{
        font-family: 'DM Mono', monospace; font-size: 0.7rem;
        text-transform: uppercase; letter-spacing: 0.12em; color: {TERRACOTTA};
    }}
    .stepper-track {{
        display: flex; align-items: flex-start;
        gap: 0; overflow-x: auto; padding-bottom: 0;
    }}
    .stepper-item {{
        display: flex; flex-direction: column;
        align-items: center; flex: 1; min-width: 80px;
        position: relative;
    }}
    .stepper-item:not(:last-child)::after {{
        content: '';
        position: absolute;
        top: 18px; left: 60%;
        width: calc(80% - 0px); height: 2px;
        background: rgba(255,255,255,0.1);
        z-index: 0;
    }}
    .stepper-item.done:not(:last-child)::after {{ background: {TERRACOTTA}; opacity: 0.6; }}
    .stepper-circle {{
        width: 36px; height: 36px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.9rem; font-weight: 700; z-index: 1;
        border: 2px solid rgba(255,255,255,0.1);
        background: rgba(255,255,255,0.04);
        color: rgba(247,242,232,0.3);
        font-family: 'DM Mono', monospace;
        transition: all 0.2s ease;
        margin-bottom: 8px;
    }}
    .stepper-item.done .stepper-circle {{
        background: rgba(196,102,58,0.2);
        border-color: {TERRACOTTA};
        color: {TERRACOTTA};
    }}
    .stepper-item.active .stepper-circle {{
        background: {TERRACOTTA};
        border-color: {GOLD};
        color: {CREAM};
        box-shadow: 0 0 0 4px rgba(196,102,58,0.2);
    }}
    .stepper-label {{
        font-family: 'DM Mono', monospace; font-size: 0.6rem;
        text-transform: uppercase; letter-spacing: 0.08em;
        color: rgba(247,242,232,0.3); text-align: center;
        max-width: 70px;
    }}
    .stepper-item.active .stepper-label {{ color: {GOLD}; font-weight: 700; }}
    .stepper-item.done .stepper-label {{ color: rgba(247,242,232,0.55); }}

    /* Pipeline step content */
    .pipeline-content {{
        padding: 40px 48px;
        min-height: 400px;
    }}
    .pipe-step-header {{
        display: flex; align-items: flex-start;
        gap: 16px; margin-bottom: 32px;
    }}
    .pipe-step-icon {{
        font-size: 2.4rem; line-height: 1;
        flex-shrink: 0;
    }}
    .pipe-step-num {{
        font-family: 'DM Mono', monospace; font-size: 0.68rem;
        text-transform: uppercase; letter-spacing: 0.12em;
        color: {TERRACOTTA}; margin-bottom: 4px;
    }}
    .pipe-step-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem; font-weight: 700; color: {CHARCOAL};
        margin-bottom: 6px; line-height: 1.2;
    }}
    .pipe-step-desc {{
        font-size: 0.9rem; color: {SLATE}; line-height: 1.7;
    }}

    /* Problem type cards */
    .prob-card {{
        padding: 32px 28px; border-radius: 20px;
        border: 2px solid {CREAM_DK};
        background: {PARCHMENT};
        cursor: pointer; text-align: center;
        transition: all 0.25s ease;
        height: 100%;
    }}
    .prob-card:hover {{ transform: translateY(-4px); box-shadow: 0 12px 40px rgba(107,29,58,0.15); }}
    .prob-card.selected {{
        border-color: {TERRACOTTA};
        background: rgba(196,102,58,0.06);
        box-shadow: 0 0 0 4px rgba(196,102,58,0.1);
    }}
    .prob-icon {{ font-size: 3rem; margin-bottom: 14px; display: block; }}
    .prob-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem; font-weight: 700; color: {CHARCOAL};
        margin-bottom: 10px;
    }}
    .prob-desc {{ font-size: 0.86rem; color: {SLATE}; line-height: 1.6; }}
    .prob-models {{
        margin-top: 14px; display: flex; flex-wrap: wrap;
        gap: 5px; justify-content: center;
    }}
    .prob-tag {{
        padding: 2px 9px; background: rgba(107,29,58,0.08);
        border: 1px solid rgba(107,29,58,0.15);
        border-radius: 12px; font-size: 0.65rem;
        font-family: 'DM Mono', monospace; color: {BURGUNDY};
    }}

    /* Data cards */
    .data-stat-card {{
        background: {PARCHMENT}; border: 1.5px solid {CREAM_DK};
        border-radius: 14px; padding: 18px 16px; text-align: center;
    }}
    .data-stat-val {{
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem; font-weight: 700; color: {BURGUNDY};
    }}
    .data-stat-lbl {{
        font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.08em; color: {MUTED}; font-weight: 700;
        margin-top: 4px;
    }}

    /* Result highlight */
    .result-box {{
        background: linear-gradient(135deg, {PARCHMENT}, {CREAM});
        border: 1.5px solid {CREAM_DK};
        border-left: 4px solid {BURGUNDY};
        border-radius: 14px; padding: 20px 18px;
        margin-bottom: 14px;
    }}
    .result-box-success {{
        border-left-color: {SAGE};
        background: linear-gradient(135deg, #EBF3EC, {CREAM});
    }}
    .result-box-warning {{
        border-left-color: {AMBER};
        background: linear-gradient(135deg, #FFF3DC, {CREAM});
    }}
    .result-box-danger {{
        border-left-color: {ANGRY_RED};
        background: linear-gradient(135deg, #FBE9E7, {CREAM});
    }}
    .result-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1rem; font-weight: 700; color: {CHARCOAL}; margin-bottom: 6px;
    }}
    .result-body {{ font-size: 0.85rem; color: {SLATE}; line-height: 1.6; }}

    /* Table */
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

    /* Distribution bars */
    .dist-row {{
        display: grid; grid-template-columns: 90px 1fr 58px 46px;
        align-items: center; gap: 12px; margin-bottom: 12px;
    }}
    .dist-n {{ font-family: 'Playfair Display', serif; font-size: 0.88rem; font-weight: 700; color: {CHARCOAL}; }}
    .dist-t {{ height: 10px; background: {CREAM_DK}; border-radius: 5px; overflow: hidden; }}
    .dist-f {{ height: 100%; border-radius: 5px; }}
    .dist-c {{ font-family: 'DM Mono', monospace; font-size: 0.76rem; color: {MUTED}; }}
    .dist-p {{ font-family: 'DM Mono', monospace; font-size: 0.8rem; font-weight: 600; color: {CHARCOAL}; text-align: right; }}

    /* Pipe cards */
    .pipe-card {{
        padding: 22px 20px; background: {PARCHMENT};
        border: 1.5px solid {CREAM_DK}; border-radius: 16px; margin-bottom: 14px;
    }}
    .pipe-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }}
    .pipe-ic {{ font-size: 1.4rem; }}
    .pipe-t {{ font-family: 'Playfair Display', serif; font-size: 0.95rem; font-weight: 700; color: {CHARCOAL}; }}
    .pipe-li {{
        font-size: 0.8rem; color: {SLATE};
        padding-left: 14px; position: relative; line-height: 1.6; margin-bottom: 5px;
    }}

    /* Model selection cards */
    .model-card {{
        padding: 24px 20px; background: {PARCHMENT};
        border: 2px solid {CREAM_DK}; border-radius: 16px;
        cursor: pointer; transition: all 0.2s ease; height: 100%;
    }}
    .model-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 28px rgba(107,29,58,0.12); }}
    .model-card.selected {{
        border-color: {TERRACOTTA};
        background: rgba(196,102,58,0.06);
        box-shadow: 0 0 0 3px rgba(196,102,58,0.12);
    }}
    .model-icon {{ font-size: 1.8rem; margin-bottom: 10px; display: block; }}
    .model-name {{
        font-family: 'Playfair Display', serif;
        font-size: 1rem; font-weight: 700; color: {CHARCOAL}; margin-bottom: 6px;
    }}
    .model-desc {{ font-size: 0.78rem; color: {SLATE}; line-height: 1.5; }}
    .model-badge {{
        display: inline-block; padding: 2px 8px;
        background: rgba(107,29,58,0.08);
        border: 1px solid rgba(107,29,58,0.15);
        border-radius: 12px; font-size: 0.6rem;
        font-family: 'DM Mono', monospace; color: {BURGUNDY};
        margin-top: 8px; margin-right: 4px;
    }}

    /* Fold results */
    .fold-bar-container {{
        display: flex; align-items: center; gap: 10px; margin-bottom: 10px;
    }}
    .fold-label {{
        font-family: 'DM Mono', monospace; font-size: 0.75rem;
        color: {SLATE}; min-width: 50px;
    }}
    .fold-track {{
        flex: 1; height: 8px; background: {CREAM_DK}; border-radius: 4px; overflow: hidden;
    }}
    .fold-fill {{ height: 100%; border-radius: 4px; }}
    .fold-val {{
        font-family: 'DM Mono', monospace; font-size: 0.75rem;
        color: {CHARCOAL}; font-weight: 700; min-width: 55px; text-align: right;
    }}

    /* Overfitting indicator */
    .overfit-indicator {{
        display: flex; align-items: center; gap: 16px;
        padding: 16px 20px; border-radius: 12px;
        margin-bottom: 14px;
    }}
    .overfit-dot {{
        width: 16px; height: 16px; border-radius: 50%; flex-shrink: 0;
    }}
    .overfit-text {{
        font-family: 'Playfair Display', serif;
        font-size: 0.95rem; font-weight: 700;
    }}
    .overfit-sub {{ font-size: 0.8rem; color: {SLATE}; margin-top: 3px; }}

    /* Tech rows */
    .tech-row {{
        display: flex; align-items: flex-start; gap: 10px;
        padding: 12px 14px; background: {CREAM};
        border-radius: 10px; border: 1.5px solid {CREAM_DK}; margin-bottom: 10px;
    }}
    .tech-l {{
        font-size: 0.62rem; text-transform: uppercase;
        letter-spacing: 0.08em; color: {MUTED}; font-weight: 700; margin-bottom: 2px;
    }}
    .tech-n {{ font-size: 0.78rem; color: {CHARCOAL}; font-weight: 600; }}

    /* Sidebar */
    .sb-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem; font-weight: 700; color: {GOLD};
        padding: 20px 20px 8px;
    }}
    .sb-divider {{
        border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 8px 20px 16px;
    }}
    .sb-section-lbl {{
        font-family: 'DM Mono', monospace; font-size: 0.62rem;
        text-transform: uppercase; letter-spacing: 0.12em;
        color: rgba(247,242,232,0.3); padding: 8px 20px 4px;
    }}

    /* Buttons */
    div[data-testid="stButton"] > button {{
        background: {CHARCOAL} !important;
        border: 2px solid {TERRACOTTA} !important;
        color: {CREAM} !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 0.95rem !important; font-style: italic !important;
        font-weight: 700 !important; border-radius: 10px !important;
        padding: 10px 24px !important;
        width: 100%; transition: all 0.2s ease !important;
    }}
    div[data-testid="stButton"] > button:hover {{
        background: {BURGUNDY} !important;
        border-color: {GOLD} !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(107,29,58,0.3) !important;
    }}
    div[data-testid="stRadio"] > label {{
        font-family: 'Lato', sans-serif !important; font-size: 0.9rem !important;
    }}

    /* CTA */
    .mf-cta {{ background: {TERRACOTTA}; padding: 60px 48px; text-align: center; }}
    .mf-cta-title {{
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem; font-weight: 700; color: {CREAM}; margin-bottom: 10px;
    }}
    .mf-cta-sub {{ font-size: 1rem; color: rgba(247,242,232,0.75); }}

    /* Footer */
    .mf-footer {{
        background: {CHARCOAL}; padding: 22px 48px;
        display: flex; align-items: center;
        justify-content: space-between; gap: 12px; flex-wrap: wrap;
        border-top: 1px solid rgba(255,255,255,0.06);
    }}
    .mf-ft-brand {{
        font-family: 'Playfair Display', serif; font-size: 1rem; color: {GOLD}; font-weight: 600;
    }}
    .mf-ft-txt {{ font-family: 'DM Mono', monospace; font-size: 0.74rem; color: {SLATE}; }}

    .appview-container .main .block-container {{ padding-top: 0 !important; }}

    /* Outlier cards */
    .outlier-method {{
        padding: 16px 18px; background: {PARCHMENT};
        border: 1.5px solid {CREAM_DK}; border-radius: 12px;
        margin-bottom: 10px;
    }}
    .outlier-method-name {{
        font-family: 'Playfair Display', serif; font-size: 0.9rem; font-weight: 700; color: {CHARCOAL};
        margin-bottom: 4px;
    }}
    .outlier-count {{
        font-family: 'DM Mono', monospace; font-size: 1.2rem; font-weight: 700; color: {BURGUNDY};
    }}
    .outlier-sub {{ font-size: 0.74rem; color: {MUTED}; }}

    /* Feature importance bars */
    .feat-imp-row {{
        display: grid; grid-template-columns: 140px 1fr 60px;
        align-items: center; gap: 10px; margin-bottom: 10px;
    }}
    .feat-imp-name {{
        font-family: 'DM Mono', monospace; font-size: 0.76rem;
        color: {CHARCOAL}; white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis;
    }}
    .feat-imp-track {{ height: 8px; background: {CREAM_DK}; border-radius: 4px; overflow: hidden; }}
    .feat-imp-fill {{ height: 100%; border-radius: 4px; }}
    .feat-imp-val {{
        font-family: 'DM Mono', monospace; font-size: 0.76rem;
        color: {SLATE}; text-align: right;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "home",
        "sel_mood": "Happy",
        # Pipeline
        "pipeline_step": 0,
        "problem_type": None,
        "df_raw": None,
        "df": None,
        "target_col": None,
        "feature_cols": [],
        "df_cleaned": None,
        "selected_features": None,
        "X_train": None, "X_test": None,
        "y_train": None, "y_test": None,
        "model": None, "model_name": None,
        "model_params": {},
        "cv_scores": None,
        "k_folds": 5,
        "test_size": 0.2,
        "train_score": None, "test_score": None,
        "outlier_mask": None,
        "scaler_fitted": None,
        "label_encoders": {},
        "tuned_model": None,
        "tuning_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — MoodFlix
# ─────────────────────────────────────────────────────────────────────────────
def render_stars(rating: float) -> str:
    filled = round((rating / 10) * 5)
    return "".join(
        f'<span style="color:{GOLD};font-size:0.95rem">★</span>' if i < filled
        else f'<span style="color:{CREAM_DK};font-size:0.95rem">★</span>'
        for i in range(5)
    ) + f' <span style="font-family:DM Mono,monospace;font-size:0.8rem;color:{SLATE};margin-left:4px">{rating}/10</span>'

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
    return f"""
    <div class="mc-card" style="background:{mc['bg']};margin-bottom:16px">
        <div style="position:absolute;left:0;top:0;bottom:0;width:4px;background:{mc['accent']};border-radius:4px 0 0 4px"></div>
        <div class="mc-rank">#{idx+1}</div>
        <div class="mc-title">{movie.get('title','')}</div>
        <div class="mc-genres">{genre_chips}</div>
        <div style="margin-bottom:10px">{stars_html}</div>
        <p class="mc-overview">{movie.get('overview','')[:200]}…</p>
        <div class="mc-footer">
            <div class="mc-meta"><span class="mc-ml">Mood</span><span class="mc-mv">{movie.get('mood','')}</span></div>
            <div class="mc-meta"><span class="mc-ml">Rating</span><span class="mc-mv">{movie.get('rating',0)}/10</span></div>
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
# HELPERS — Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def render_stepper(current_step):
    items = ""
    for i, (icon, label) in enumerate(PIPELINE_STEPS):
        if i < current_step:
            cls = "done"
            circle_content = "✓"
        elif i == current_step:
            cls = "active"
            circle_content = icon
        else:
            cls = ""
            circle_content = str(i)
        items += f"""
        <div class="stepper-item {cls}">
            <div class="stepper-circle">{circle_content}</div>
            <span class="stepper-label">{label}</span>
        </div>"""
    return f"""
    <div class="pipeline-stepper">
        <div class="pipeline-header">
            <div>
                <div class="pipeline-subtitle">Interactive ML Workflow</div>
                <div class="pipeline-title">ML Pipeline Lab</div>
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:rgba(247,242,232,0.4)">
                Step {current_step + 1} of {len(PIPELINE_STEPS)}
            </div>
        </div>
        <div class="stepper-track">{items}</div>
        <div style="height:24px"></div>
    </div>"""

def step_header(icon, num, title, desc):
    st.markdown(f"""
    <div class="pipe-step-header">
        <div class="pipe-step-icon">{icon}</div>
        <div>
            <div class="pipe-step-num">Step {num} · {title}</div>
            <div class="pipe-step-title">{title}</div>
            <div class="pipe-step-desc">{desc}</div>
        </div>
    </div>""", unsafe_allow_html=True)

def nav_buttons(back_label="← Back", next_label="Next →", show_back=True):
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if show_back and st.button(back_label, key=f"back_{st.session_state.pipeline_step}"):
            st.session_state.pipeline_step = max(0, st.session_state.pipeline_step - 1)
            st.rerun()
    with col3:
        if st.button(next_label, key=f"next_{st.session_state.pipeline_step}"):
            st.session_state.pipeline_step = min(len(PIPELINE_STEPS)-1, st.session_state.pipeline_step + 1)
            st.rerun()

def data_stat_card(val, label):
    return f"""<div class="data-stat-card">
        <div class="data-stat-val">{val}</div>
        <div class="data-stat-lbl">{label}</div>
    </div>"""

def result_box(title, body, kind="default"):
    cls = {"success":"result-box-success","warning":"result-box-warning","danger":"result-box-danger"}.get(kind,"")
    return f"""<div class="result-box {cls}">
        <div class="result-title">{title}</div>
        <div class="result-body">{body}</div>
    </div>"""

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div class="sb-title">🎬 MoodFlix ✦ ML Lab</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-lbl">MoodFlix</div>', unsafe_allow_html=True)

    for icon, label, pid in [("🏠","Home","home"),("🎬","Get Recs","rec"),("📊","ML Metrics","met"),("ℹ️","About","abt")]:
        if st.button(f"{icon}  {label}", key=f"nav_{pid}", use_container_width=True):
            st.session_state.page = pid
            st.rerun()

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-lbl">ML Pipeline Lab</div>', unsafe_allow_html=True)

    if st.button("🔬  Open Pipeline Lab", key="nav_pipeline", use_container_width=True):
        st.session_state.page = "pipeline"
        st.rerun()

    if st.session_state.page == "pipeline":
        for i, (icon, label) in enumerate(PIPELINE_STEPS):
            done = i < st.session_state.pipeline_step
            active = i == st.session_state.pipeline_step
            color = TERRACOTTA if active else (SAGE if done else "rgba(247,242,232,0.3)")
            prefix = "✓ " if done else ("▶ " if active else "  ")
            if st.button(f"{prefix}{icon} {label}", key=f"nav_pipe_{i}", use_container_width=True):
                st.session_state.pipeline_step = i
                st.rerun()

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sb-section-lbl">Status</div>', unsafe_allow_html=True)

    if MOODFLIX_OK:
        st.markdown(f'<div style="padding:6px 20px;font-size:0.76rem;color:{SAGE}">✅ MoodFlix models ready</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="padding:6px 20px;font-size:0.76rem;color:{TERRACOTTA}">⚠️ Run train.py first</div>', unsafe_allow_html=True)

    if st.session_state.df is not None:
        shape = st.session_state.df.shape
        st.markdown(f'<div style="padding:6px 20px;font-size:0.76rem;color:{SAGE}">✅ Data: {shape[0]}×{shape[1]}</div>', unsafe_allow_html=True)

    if st.session_state.model is not None:
        st.markdown(f'<div style="padding:6px 20px;font-size:0.76rem;color:{SAGE}">✅ Model: {st.session_state.model_name}</div>', unsafe_allow_html=True)

    if SKLEARN_OK:
        st.markdown(f'<div style="padding:6px 20px;font-size:0.76rem;color:{SAGE}">✅ sklearn ready</div>', unsafe_allow_html=True)

    if not PLOTLY_OK:
        st.markdown(f'<div style="padding:6px 20px;font-size:0.76rem;color:{AMBER}">⚠️ Install plotly</div>', unsafe_allow_html=True)

    if st.session_state.page == "pipeline" and st.session_state.pipeline_step > 0:
        st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
        if st.button("🔄  Reset Pipeline", key="reset_pipe", use_container_width=True):
            for k in ["pipeline_step","problem_type","df_raw","df","target_col","feature_cols",
                      "df_cleaned","selected_features","X_train","X_test","y_train","y_test",
                      "model","model_name","model_params","cv_scores","k_folds","test_size",
                      "train_score","test_score","outlier_mask","scaler_fitted","label_encoders",
                      "tuned_model","tuning_results"]:
                st.session_state[k] = None if k not in ["pipeline_step","k_folds","test_size","model_params","feature_cols","label_encoders"] else (0 if k=="pipeline_step" else (5 if k=="k_folds" else (0.2 if k=="test_size" else ({} if k in ["model_params","label_encoders"] else []))))
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    st.markdown(f"""
    <div class="mf-hero">
        <div class="mf-hero-bg">CINEMA</div>
        <p class="mf-eyebrow">MoodFlix ✦ ML Pipeline Lab · TMDb Dataset · scikit-learn</p>
        <h1 class="mf-hero-title">Your Mood,<br><em>Your Pipeline.</em></h1>
        <p class="mf-hero-desc">
            A cinematic fusion — emotion-based movie recommendations powered by
            SVM & K-Means, plus a full interactive 9-step ML pipeline that works on
            <em>any</em> CSV dataset. Pick your mood or build your own model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{BURGUNDY};padding:22px 48px;display:flex;gap:60px;flex-wrap:wrap">
        <div class="mf-stat"><span class="mf-stat-v">92.88%</span><span class="mf-stat-l">SVM Accuracy</span></div>
        <div class="mf-stat"><span class="mf-stat-v">4,775</span><span class="mf-stat-l">Movies indexed</span></div>
        <div class="mf-stat"><span class="mf-stat-v">9</span><span class="mf-stat-l">Pipeline Steps</span></div>
        <div class="mf-stat"><span class="mf-stat-v">Any CSV</span><span class="mf-stat-l">Custom datasets</span></div>
        <div class="mf-stat"><span class="mf-stat-v">6+</span><span class="mf-stat-l">ML Models</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="mf-section">
        <div style="text-align:center;margin-bottom:36px">
            <span class="mf-sec-ey">Two Experiences in One</span>
            <div class="mf-sec-title" style="text-align:center">What Would You Like to Do?</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div style="background:{PARCHMENT};border:2px solid {CREAM_DK};border-radius:20px;padding:36px;height:100%">
            <div style="font-size:3rem;margin-bottom:16px">🎬</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:700;color:{CHARCOAL};margin-bottom:12px">
                MoodFlix Recommender
            </div>
            <p style="font-size:0.9rem;color:{SLATE};line-height:1.7;margin-bottom:20px">
                Tell us how you're feeling. Our SVM classifier and K-Means clustering
                pipeline surface the perfect films from 4,775 TMDb movies, matched to
                your emotional state in real-time.
            </p>
            <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px">
                {''.join(f'<span style="padding:3px 10px;background:rgba(107,29,58,0.08);border:1px solid rgba(107,29,58,0.15);border-radius:12px;font-size:0.65rem;font-family:DM Mono,monospace;color:{BURGUNDY}">{t}</span>' for t in ["SVM", "K-Means", "PCA", "TMDb", "5 Moods"])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("✨  Get Movie Recommendations", key="cta_rec"):
            st.session_state.page = "rec"
            st.rerun()

    with col_b:
        st.markdown(f"""
        <div style="background:{CHARCOAL};border:2px solid rgba(212,175,55,0.2);border-radius:20px;padding:36px;height:100%">
            <div style="font-size:3rem;margin-bottom:16px">🔬</div>
            <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:700;color:{CREAM};margin-bottom:12px">
                ML Pipeline Lab
            </div>
            <p style="font-size:0.9rem;color:rgba(247,242,232,0.6);line-height:1.7;margin-bottom:20px">
                Upload any CSV and walk through a full 9-step ML workflow:
                EDA, data cleaning, outlier detection, feature selection,
                model training, KFold CV, performance metrics, and hyperparameter tuning.
            </p>
            <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px">
                {''.join(f'<span style="padding:3px 10px;background:rgba(212,175,55,0.1);border:1px solid rgba(212,175,55,0.2);border-radius:12px;font-size:0.65rem;font-family:DM Mono,monospace;color:{GOLD}">{t}</span>' for t in ["Any CSV", "EDA", "IQR/IsoForest", "KFold", "GridSearch"])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        if st.button("🔬  Open ML Pipeline Lab", key="cta_pipeline"):
            st.session_state.page = "pipeline"
            st.rerun()

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="mf-section-dark">
        <div style="text-align:center;margin-bottom:36px">
            <span class="mf-sec-ey">Under the Hood</span>
            <div class="mf-sec-title mf-sec-title-lt" style="text-align:center">Pipeline Architecture</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    for col, (ic, t, b) in zip([f1,f2,f3,f4], [
        ("🧠","SVM + KNN + LR","Three classifiers with accuracy, F1, confusion matrix and comparative metrics."),
        ("🔍","K-Means + PCA","Unsupervised clustering and dimensionality reduction for both recommender and lab."),
        ("🧹","Smart Cleaning","IQR, Isolation Forest, DBSCAN, OPTICS outlier detection with interactive removal."),
        ("⚙️","Hyperparameter Lab","GridSearch & RandomSearch tuning with before/after performance visualization."),
    ]):
        with col:
            st.markdown(f"""<div class="feat-card">
                <span class="feat-ic">{ic}</span>
                <div class="feat-t">{t}</div>
                <div class="feat-b">{b}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RECOMMEND
# ─────────────────────────────────────────────────────────────────────────────
def page_rec():
    st.markdown(f"""
    <div style="background:{CHARCOAL};padding:52px 48px 44px;border-bottom:3px solid {BURGUNDY}">
        <p class="mf-eyebrow">Powered by SVM + K-Means · TMDb 5000</p>
        <h1 style="font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:700;color:{CREAM};margin-bottom:10px">
            How are you feeling?
        </h1>
        <p style="font-size:0.95rem;color:rgba(247,242,232,0.58)">
            Choose your current mood and let our ML pipeline find the perfect film.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    st.markdown(f"""<div class="mf-panel">
        <div class="mf-panel-title"><span class="mf-pnum">01</span> Select Your Mood</div>
    </div>""", unsafe_allow_html=True)

    moods_data = [
        ("😄","Happy","Comedy · Feel-good",TERRACOTTA,"#FFF3DC"),
        ("😢","Sad","Drama · Emotional",SAD_BLUE,"#EDF0F7"),
        ("😤","Angry","Action · Adrenaline",ANGRY_RED,"#FBE9E7"),
        ("😌","Relaxed","Romance · Heartwarming",RELAX_GRN,"#EBF3EC"),
        ("😐","Neutral","Mixed · Any genre",NEUTRAL_PRP,"#F5F0F8"),
    ]

    cols = st.columns(5)
    for col, (em, label, sub, color, bg) in zip(cols, moods_data):
        with col:
            selected = st.session_state.sel_mood == label
            border = f"2px solid {color}" if selected else "2px solid transparent"
            transform = "translateY(-3px)" if selected else "none"
            shadow = "0 6px 20px rgba(0,0,0,0.12)" if selected else "none"
            st.markdown(f"""
            <div class="mood-tile {'selected' if selected else ''}"
                 style="background:{bg};border:{border};transform:{transform};box-shadow:{shadow}">
                <span class="mood-em">{em}</span>
                <span class="mood-lb" style="color:{color}">{label}</span>
                <span class="mood-sb">{sub}</span>
            </div>""", unsafe_allow_html=True)
            if st.button(label, key=f"mood_{label}", use_container_width=True):
                st.session_state.sel_mood = label
                st.rerun()

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown(f"""<div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:24px 22px">
            <div class="mf-panel-title"><span class="mf-pnum">02</span> ML Model</div>
        </div>""", unsafe_allow_html=True)
        model_map = {
            "SVM  —  Primary · Best accuracy": "svm",
            "KNN  —  K-Nearest Neighbours": "knn",
            "Logistic Regression  —  Baseline": "lr",
        }
        sel_model_label = st.radio("model_pick", list(model_map.keys()), label_visibility="collapsed")
        sel_model = model_map[sel_model_label]

    with right_col:
        st.markdown(f"""<div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:24px 22px">
            <div class="mf-panel-title"><span class="mf-pnum">03</span> Number of Results</div>
        </div>""", unsafe_allow_html=True)
        top_n = st.select_slider("top_n_slider", options=[3, 5, 8, 10], value=5, label_visibility="collapsed")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button(f'✨ Get Recommendations for "{st.session_state.sel_mood}" →', use_container_width=True):
        if not MOODFLIX_OK:
            st.error(f"⚠️ MoodFlix backend not available. Run `python train.py` first in the movie_recommender folder.")
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
            SVM, KNN and Logistic Regression — evaluated on the TMDb 5000 dataset.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    if MOODFLIX_OK:
        metrics = get_metrics()
    else:
        metrics = {
            "SVM": {"accuracy":0.9288,"precision":0.9287,"recall":0.9288,"f1":0.9244},
            "KNN": {"accuracy":0.9141,"precision":0.9110,"recall":0.9141,"f1":0.9088},
            "Logistic Regression": {"accuracy":0.9288,"precision":0.9311,"recall":0.9288,"f1":0.9241},
        }

    best = max(metrics, key=lambda m: metrics[m]["f1"])
    mc1,mc2,mc3,mc4 = st.columns(4)
    for col, (lbl, val) in zip([mc1,mc2,mc3,mc4], [
        ("Best Model", best),
        ("SVM Accuracy", f"{metrics['SVM']['accuracy']*100:.2f}%"),
        ("SVM F1 Score", f"{metrics['SVM']['f1']*100:.2f}%"),
        ("Dataset size", "4,775"),
    ]):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-lbl">{lbl}</div>
                <div class="metric-val">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    if PLOTLY_OK:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy by Model", "F1 Score by Model"))
        model_names = list(metrics.keys())
        colors = [BURGUNDY, SAGE, TERRACOTTA]
        for i, m in enumerate(model_names):
            fig.add_trace(go.Bar(name=m, x=[m], y=[metrics[m]["accuracy"]*100],
                marker_color=colors[i], showlegend=False), row=1, col=1)
            fig.add_trace(go.Bar(name=m, x=[m], y=[metrics[m]["f1"]*100],
                marker_color=colors[i], showlegend=i==0), row=1, col=2)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Mono", color=CHARCOAL, size=11),
            height=320, margin=dict(l=20, r=20, t=40, b=20),
            bargap=0.3,
        )
        fig.update_yaxes(range=[85, 95], ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        bl, br = st.columns(2)
        with bl:
            st.markdown(f"""
            <div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:26px 24px">
                <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{CHARCOAL};margin-bottom:20px">Accuracy by Model</div>
                {bar_row_html('SVM ⭐', metrics['SVM']['accuracy']*100, BURGUNDY, f"{metrics['SVM']['accuracy']*100:.2f}%")}
                {bar_row_html('Log. Reg.', metrics['Logistic Regression']['accuracy']*100, SAGE, f"{metrics['Logistic Regression']['accuracy']*100:.2f}%")}
                {bar_row_html('KNN', metrics['KNN']['accuracy']*100, TERRACOTTA, f"{metrics['KNN']['accuracy']*100:.2f}%")}
            </div>""", unsafe_allow_html=True)
        with br:
            st.markdown(f"""
            <div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:26px 24px">
                <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{CHARCOAL};margin-bottom:20px">F1 Score by Model</div>
                {bar_row_html('SVM ⭐', metrics['SVM']['f1']*100, BURGUNDY, f"{metrics['SVM']['f1']*100:.2f}%")}
                {bar_row_html('Log. Reg.', metrics['Logistic Regression']['f1']*100, SAGE, f"{metrics['Logistic Regression']['f1']*100:.2f}%")}
                {bar_row_html('KNN', metrics['KNN']['f1']*100, TERRACOTTA, f"{metrics['KNN']['f1']*100:.2f}%")}
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    def _srows(met, b):
        out = ""
        for m in met:
            rc = "best-row" if m == b else ""
            badge = f'<span class="best-badge">★ Best</span> ' if m == b else ""
            fc = "f1-cell" if m == b else ""
            out += (f'<tr class="{rc}"><td><div>{badge}{m}</div></td>'
                    + f'<td>{met[m]["accuracy"]*100:.2f}%</td>'
                    + f'<td>{met[m]["precision"]*100:.2f}%</td>'
                    + f'<td>{met[m]["recall"]*100:.2f}%</td>'
                    + f'<td class="{fc}">{met[m]["f1"]*100:.2f}%</td></tr>')
        return out

    st.markdown(f"""
    <div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:26px 24px">
        <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{CHARCOAL};margin-bottom:20px">Model Scorecard</div>
        <div style="overflow-x:auto"><table class="mf-table">
        <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th></tr></thead>
        <tbody>{_srows(metrics, best)}</tbody></table></div></div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:26px 24px">
        <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{CHARCOAL};margin-bottom:20px">Mood Label Distribution</div>
        {"".join(f'''<div class="dist-row">
            <span class="dist-n">{n}</span>
            <div class="dist-t"><div class="dist-f" style="width:{p}%;background:{c}"></div></div>
            <span class="dist-c">{cnt:,}</span>
            <span class="dist-p">{p}%</span>
        </div>''' for n,cnt,p,c in [
            ("Sad",1602,33.6,SAD_BLUE),("Happy",1299,27.2,TERRACOTTA),
            ("Angry",1034,21.7,ANGRY_RED),("Neutral",718,15.0,NEUTRAL_PRP),("Relaxed",122,2.6,RELAX_GRN)
        ])}
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
def page_about():
    st.markdown(f"""
    <div style="background:{CHARCOAL};padding:52px 48px 44px;border-bottom:3px solid {GOLD}">
        <p class="mf-eyebrow" style="color:{GOLD}">Academic ML Project · Combined Dashboard</p>
        <h1 style="font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:700;color:{CREAM};margin-bottom:10px">
            About This App
        </h1>
        <p style="font-size:0.95rem;color:rgba(247,242,232,0.58)">
            MoodFlix Recommender meets a full interactive ML Pipeline Lab.
        </p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{PARCHMENT};border:1.5px solid {CREAM_DK};border-radius:18px;padding:28px 24px;margin-bottom:20px">
        <div style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;color:{CHARCOAL};margin-bottom:12px;padding-bottom:10px;border-bottom:1.5px solid {CREAM_DK}">
            Project Overview
        </div>
        <p style="font-size:0.9rem;color:{SLATE};line-height:1.75;margin-bottom:12px">
            This combined application merges two ML projects: the <strong>MoodFlix</strong> emotion-based movie recommender
            (SVM, K-Means, PCA on TMDb 5000) and a full generic <strong>ML Pipeline Lab</strong> that accepts any CSV
            dataset and guides users through a complete 9-step workflow.
        </p>
        <p style="font-size:0.9rem;color:{SLATE};line-height:1.75">
            The pipeline covers every academic ML component: problem framing, EDA, data engineering, outlier detection
            (IQR, Isolation Forest, DBSCAN, OPTICS), feature selection, train/test split, model training with KFold CV,
            performance metrics with overfit/underfit detection, and hyperparameter tuning (GridSearch / RandomSearch).
        </p>
    </div>""", unsafe_allow_html=True)

    t1,t2,t3,t4 = st.columns(4)
    for col, (ic, layer, name) in zip([t1,t2,t3,t4]*2, [
        ("🎬","Dataset","TMDb 5000 Movies"),("🧹","Preprocessing","Pandas · StandardScaler"),
        ("📐","Feature Eng.","PCA (scikit-learn)"),("🤖","Classification","SVM · KNN · LogReg"),
        ("🔍","Clustering","K-Means (K=5)"),("🔬","ML Lab","Any CSV · 9 Steps"),
        ("⚙️","Tuning","GridSearch · RandomSearch"),("🖥️","Frontend","Streamlit · Plotly"),
    ]):
        with col:
            st.markdown(f"""<div class="tech-row" style="margin-bottom:10px">
                <span style="font-size:1.2rem">{ic}</span>
                <div><div class="tech-l">{layer}</div><div class="tech-n">{name}</div></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{CHARCOAL};border-radius:18px;padding:28px 24px">
        <div style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;color:{GOLD};margin-bottom:16px">
            Academic Components Demonstrated
        </div>
        {"".join(f'''<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
            <span style="color:{SAGE};font-size:1rem">✅</span>
            <span style="font-size:0.88rem;color:rgba(247,242,232,0.8)">{item}</span>
        </div>''' for item in [
            "Data preprocessing — missing values, encoding, normalisation, type inference",
            "Feature engineering — PCA with variance analysis and interactive 2D/3D scatter",
            "Exploratory Data Analysis — distributions, correlations, missing value heatmap",
            "Data cleaning — Mean/Median/Mode imputation + IQR/IsolationForest/DBSCAN/OPTICS outliers",
            "Feature selection — Variance threshold, correlation filter, mutual information / info gain",
            "Classification — SVM (RBF/linear/poly kernel), Random Forest, KNN, Logistic Regression",
            "Regression — Linear Regression, SVR, Random Forest Regressor",
            "Model evaluation — Accuracy, Precision, Recall, F1, MAE, MSE, RMSE, R², Confusion Matrix",
            "KFold / Stratified KFold cross-validation with per-fold score visualisation",
            "Overfitting / underfitting detection via train vs test gap analysis",
            "Hyperparameter tuning — GridSearchCV and RandomizedSearchCV with before/after comparison",
            "MoodFlix recommender — SVM + K-Means pipeline on TMDb 5000 dataset",
        ])}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 0: PROBLEM TYPE
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_step_0():
    step_header("🎯", 1, "Problem Type", "Define what kind of ML problem you want to solve. This determines available models and evaluation metrics.")

    col_a, col_b = st.columns(2)

    with col_a:
        selected_a = st.session_state.problem_type == "Classification"
        st.markdown(f"""
        <div class="prob-card {'selected' if selected_a else ''}">
            <span class="prob-icon">🏷️</span>
            <div class="prob-title">Classification</div>
            <div class="prob-desc">
                Predict discrete class labels. Ideal for binary or multi-class outcomes —
                spam detection, disease diagnosis, sentiment analysis, and more.
            </div>
            <div class="prob-models">
                {''.join(f'<span class="prob-tag">{m}</span>' for m in ["SVM","Random Forest","KNN","Logistic Reg"])}
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("✅  Select Classification", key="prob_clf", use_container_width=True):
            st.session_state.problem_type = "Classification"
            st.rerun()

    with col_b:
        selected_b = st.session_state.problem_type == "Regression"
        st.markdown(f"""
        <div class="prob-card {'selected' if selected_b else ''}">
            <span class="prob-icon">📈</span>
            <div class="prob-title">Regression</div>
            <div class="prob-desc">
                Predict continuous numerical values. Ideal for forecasting prices,
                temperatures, scores, or any real-valued output from your data.
            </div>
            <div class="prob-models">
                {''.join(f'<span class="prob-tag">{m}</span>' for m in ["Linear Reg","SVR","Random Forest","Ridge"])}
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button("✅  Select Regression", key="prob_reg", use_container_width=True):
            st.session_state.problem_type = "Regression"
            st.rerun()

    if st.session_state.problem_type:
        st.markdown(f"""
        <div class="result-box result-box-success" style="margin-top:20px">
            <div class="result-title">✓ Problem type set: {st.session_state.problem_type}</div>
            <div class="result-body">Proceed to upload your dataset in the next step.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        nav_buttons(show_back=False, next_label="Next: Upload Data →")
    else:
        st.markdown(f"""
        <div class="result-box result-box-warning" style="margin-top:20px">
            <div class="result-title">⚠️ Please select a problem type to continue</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 1: DATA INPUT
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_step_1():
    step_header("📂", 2, "Data Input", "Upload your CSV dataset, select the target variable, choose features, and explore the shape of your data with PCA.")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], key="csv_upload")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df_raw = df.copy()
            st.session_state.df = df.copy()
            st.session_state.df_cleaned = None
            st.session_state.outlier_mask = None
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if st.session_state.df is None:
        st.markdown(f"""
        <div class="result-box result-box-warning">
            <div class="result-title">⚠️ No dataset loaded</div>
            <div class="result-body">Upload a CSV file above to continue. For testing, try any classic dataset like Iris, Titanic, or California Housing.</div>
        </div>""", unsafe_allow_html=True)
        nav_buttons(back_label="← Back")
        return

    df = st.session_state.df
    n_rows, n_cols = df.shape
    n_missing = df.isnull().sum().sum()
    n_numeric = df.select_dtypes(include=np.number).columns.tolist()
    n_cat = df.select_dtypes(include=["object","category"]).columns.tolist()

    c1,c2,c3,c4 = st.columns(4)
    for col, (v, l) in zip([c1,c2,c3,c4],[
        (f"{n_rows:,}", "Rows"),
        (str(n_cols), "Columns"),
        (str(n_missing), "Missing Values"),
        (f"{len(n_numeric)}/{len(n_cat)}", "Num / Cat"),
    ]):
        with col:
            st.markdown(data_stat_card(v, l), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Select Target Column")
    target = st.selectbox("Target variable (what you want to predict)", df.columns.tolist(), key="target_select")
    st.session_state.target_col = target

    feature_options = [c for c in df.columns if c != target]
    st.subheader("Select Feature Columns")
    default_features = feature_options[:min(len(feature_options), 10)]
    selected_features = st.multiselect("Choose input features", feature_options, default=default_features, key="feature_select")
    st.session_state.feature_cols = selected_features if selected_features else feature_options

    if len(st.session_state.feature_cols) >= 2 and SKLEARN_OK and PLOTLY_OK:
        st.subheader("PCA — Data Shape Visualization")
        pca_tab, var_tab = st.tabs(["📍 PCA Scatter", "📊 Variance Explained"])

        with pca_tab:
            try:
                df_pca = df[st.session_state.feature_cols + [target]].dropna()
                X_pca = df_pca[st.session_state.feature_cols].copy()

                # Encode categoricals
                for col in X_pca.select_dtypes(include=["object","category"]).columns:
                    le = LabelEncoder()
                    X_pca[col] = le.fit_transform(X_pca[col].astype(str))

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_pca)

                n_comp = min(3, X_scaled.shape[1])
                pca = PCA(n_components=n_comp)
                comps = pca.fit_transform(X_scaled)

                pca_df = pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(n_comp)])
                pca_df["Target"] = df_pca[target].values.astype(str)

                n_unique = pca_df["Target"].nunique()
                color_seq = [BURGUNDY, TERRACOTTA, GOLD, TEAL, SAGE, SAD_BLUE, NEUTRAL_PRP, AMBER]

                if n_comp >= 3:
                    fig = px.scatter_3d(pca_df, x="PC1", y="PC2", z="PC3", color="Target",
                        title="PCA 3D — First 3 Principal Components",
                        color_discrete_sequence=color_seq[:n_unique])
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Mono", size=10),
                        height=450, margin=dict(l=0,r=0,t=40,b=0))
                else:
                    fig = px.scatter(pca_df, x="PC1", y="PC2", color="Target",
                        title="PCA 2D — First 2 Principal Components",
                        color_discrete_sequence=color_seq[:n_unique])
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="DM Mono", size=10), height=400)
                st.plotly_chart(fig, use_container_width=True)

                total_var = sum(pca.explained_variance_ratio_) * 100
                st.markdown(result_box(
                    f"✓ {n_comp} components explain {total_var:.1f}% of variance",
                    f"Data projected from {len(st.session_state.feature_cols)} dimensions to {n_comp} principal components.",
                    "success"
                ), unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"PCA visualization failed: {e}")

        with var_tab:
            try:
                max_comp = min(len(st.session_state.feature_cols), X_scaled.shape[0]-1, 20)
                pca_full = PCA(n_components=max_comp)
                pca_full.fit(X_scaled)
                cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(max_comp)],
                    y=pca_full.explained_variance_ratio_*100,
                    marker_color=TERRACOTTA, name="Individual", opacity=0.7))
                fig2.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(max_comp)],
                    y=cumvar, mode="lines+markers", name="Cumulative %",
                    line=dict(color=BURGUNDY, width=2.5), marker=dict(size=6)))
                fig2.add_hline(y=90, line_dash="dash", line_color=GOLD,
                    annotation_text="90% threshold", annotation_position="right")
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Mono", size=10), height=360,
                    yaxis_title="Variance Explained (%)", legend=dict(orientation="h"))
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.warning(f"Variance plot failed: {e}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    nav_buttons(back_label="← Back", next_label="Next: Explore Data →")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 2: EDA
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_step_2():
    step_header("🔍", 3, "Exploratory Data Analysis", "Understand your data through distributions, correlations, missing values, and summary statistics.")

    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    if df is None:
        st.warning("Please upload data in Step 2 first.")
        nav_buttons()
        return

    target = st.session_state.target_col
    features = st.session_state.feature_cols
    num_cols = df[features].select_dtypes(include=np.number).columns.tolist() if features else []
    cat_cols = df[features].select_dtypes(include=["object","category"]).columns.tolist() if features else []

    tab_summary, tab_dist, tab_corr, tab_missing = st.tabs(["📋 Summary", "📊 Distributions", "🔗 Correlations", "❓ Missing Values"])

    with tab_summary:
        st.markdown(f"""
        <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{CHARCOAL};margin-bottom:12px">
            Statistical Summary
        </div>""", unsafe_allow_html=True)
        st.dataframe(df[features + ([target] if target else [])].describe().round(3), use_container_width=True)

        if target and PLOTLY_OK:
            st.markdown(f"**Target Distribution: `{target}`**")
            if df[target].dtype in [np.float64, np.int64]:
                fig = px.histogram(df, x=target, nbins=30, color_discrete_sequence=[BURGUNDY],
                    title=f"Distribution of {target}")
            else:
                vc = df[target].value_counts().reset_index()
                vc.columns = [target, "count"]
                fig = px.bar(vc, x=target, y="count", color_discrete_sequence=[BURGUNDY],
                    title=f"Class Distribution — {target}")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Mono", size=10), height=340)
            st.plotly_chart(fig, use_container_width=True)

    with tab_dist:
        if not num_cols:
            st.info("No numeric feature columns selected.")
        elif PLOTLY_OK:
            sel_col = st.selectbox("Select column to visualise", num_cols, key="eda_dist_col")
            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x=sel_col, nbins=40, color_discrete_sequence=[TERRACOTTA],
                    title=f"Histogram — {sel_col}", marginal="box")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Mono", size=10), height=340)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.box(df, y=sel_col, color_discrete_sequence=[BURGUNDY],
                    title=f"Box Plot — {sel_col}")
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Mono", size=10), height=340)
                st.plotly_chart(fig2, use_container_width=True)

            if cat_cols:
                cat_col = st.selectbox("Categorical column", cat_cols, key="eda_cat_col")
                vc = df[cat_col].value_counts().head(20).reset_index()
                vc.columns = [cat_col, "count"]
                fig3 = px.bar(vc, x=cat_col, y="count", color_discrete_sequence=[TEAL],
                    title=f"Value Counts — {cat_col}")
                fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Mono", size=10), height=300)
                st.plotly_chart(fig3, use_container_width=True)

    with tab_corr:
        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns for correlation.")
        elif PLOTLY_OK:
            corr_df = df[num_cols].corr()
            fig = px.imshow(corr_df, color_continuous_scale=[[0, CHARCOAL],[0.5, CREAM],[1, BURGUNDY]],
                title="Correlation Heatmap", text_auto=".2f", aspect="auto")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Mono", size=9), height=450)
            st.plotly_chart(fig, use_container_width=True)

            if target and target in num_cols:
                target_corr = corr_df[target].drop(target).sort_values(key=abs, ascending=True)
                fig2 = px.bar(y=target_corr.index, x=target_corr.values,
                    orientation="h", color=target_corr.values,
                    color_continuous_scale=[[0, ANGRY_RED],[0.5, CREAM_DK],[1, SAGE]],
                    title=f"Correlation with Target: {target}")
                fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Mono", size=10), height=400, coloraxis_showscale=False)
                st.plotly_chart(fig2, use_container_width=True)

    with tab_missing:
        missing = df[features + ([target] if target else [])].isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.markdown(result_box("✓ No missing values detected", "Your dataset is complete — no imputation required.", "success"),
                unsafe_allow_html=True)
        else:
            st.markdown(result_box(f"⚠️ {missing.sum()} missing values found across {len(missing)} columns",
                f"Missing data requires imputation in the Data Engineering step.", "warning"),
                unsafe_allow_html=True)
            if PLOTLY_OK:
                fig = px.bar(x=missing.index, y=missing.values,
                    color=missing.values, color_continuous_scale=[[0, CREAM_DK],[1, BURGUNDY]],
                    title="Missing Values per Column")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Mono", size=10), height=340, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    nav_buttons(next_label="Next: Clean Data →")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 3: DATA ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_step_3():
    step_header("🔧", 4, "Data Engineering & Cleaning", "Handle missing values with imputation strategies and detect/remove outliers using multiple detection methods.")

    df_work = (st.session_state.df_cleaned if st.session_state.df_cleaned is not None
               else st.session_state.df)
    if df_work is None:
        st.warning("Please upload data first.")
        nav_buttons()
        return

    target = st.session_state.target_col
    features = st.session_state.feature_cols
    num_cols = df_work[features].select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_work[features].select_dtypes(include=["object","category"]).columns.tolist()

    # ── Imputation ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;color:{CHARCOAL};margin-bottom:16px">
        Missing Value Imputation
    </div>""", unsafe_allow_html=True)

    missing_count = df_work[features].isnull().sum().sum()
    if missing_count == 0:
        st.markdown(result_box("✓ No missing values", "Dataset is complete. Imputation not required.", "success"),
            unsafe_allow_html=True)
    else:
        st.markdown(result_box(f"⚠️ {missing_count} missing values found",
            "Select an imputation strategy below.", "warning"), unsafe_allow_html=True)

        col_imp1, col_imp2 = st.columns(2)
        with col_imp1:
            num_strategy = st.selectbox("Numeric imputation", ["Mean","Median","Mode","Zero"], key="num_impute")
        with col_imp2:
            cat_strategy = st.selectbox("Categorical imputation", ["Mode","Unknown/Missing"], key="cat_impute")

        if st.button("🔧 Apply Imputation", key="apply_impute"):
            df_imp = df_work.copy()
            for col in num_cols:
                if df_imp[col].isnull().sum() > 0:
                    if num_strategy == "Mean": df_imp[col] = df_imp[col].fillna(df_imp[col].mean())
                    elif num_strategy == "Median": df_imp[col] = df_imp[col].fillna(df_imp[col].median())
                    elif num_strategy == "Mode": df_imp[col] = df_imp[col].fillna(df_imp[col].mode()[0])
                    elif num_strategy == "Zero": df_imp[col] = df_imp[col].fillna(0)
            for col in cat_cols:
                if df_imp[col].isnull().sum() > 0:
                    if cat_strategy == "Mode": df_imp[col] = df_imp[col].fillna(df_imp[col].mode()[0])
                    else: df_imp[col] = df_imp[col].fillna("Missing")
            st.session_state.df_cleaned = df_imp
            df_work = df_imp
            st.success(f"✅ Imputation applied: {num_strategy} for numeric, {cat_strategy} for categorical")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Outlier Detection ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="font-family:'Playfair Display',serif;font-size:1.2rem;font-weight:700;color:{CHARCOAL};margin-bottom:16px">
        Outlier Detection
    </div>""", unsafe_allow_html=True)

    if not num_cols or not SKLEARN_OK:
        st.info("No numeric columns available or sklearn not installed.")
    else:
        col_od1, col_od2 = st.columns([2,1])
        with col_od1:
            outlier_methods = st.multiselect(
                "Detection methods",
                ["IQR (Interquartile Range)", "Isolation Forest", "DBSCAN", "OPTICS"],
                default=["IQR (Interquartile Range)"],
                key="outlier_methods"
            )
        with col_od2:
            contamination = st.slider("Contamination rate (for IF/DBSCAN)", 0.01, 0.3, 0.05, 0.01, key="contam")

        if st.button("🔍 Detect Outliers", key="detect_outliers") and outlier_methods:
            df_num = df_work[num_cols].dropna()
            all_outlier_masks = {}

            for method in outlier_methods:
                if "IQR" in method:
                    mask = np.zeros(len(df_num), dtype=bool)
                    for col in num_cols:
                        Q1, Q3 = df_num[col].quantile(0.25), df_num[col].quantile(0.75)
                        IQR = Q3 - Q1
                        mask |= ((df_num[col] < Q1 - 1.5*IQR) | (df_num[col] > Q3 + 1.5*IQR))
                    all_outlier_masks["IQR"] = mask

                elif "Isolation Forest" in method:
                    iso = IsolationForest(contamination=contamination, random_state=42)
                    scaler_iso = StandardScaler()
                    X_scaled = scaler_iso.fit_transform(df_num)
                    preds = iso.fit_predict(X_scaled)
                    all_outlier_masks["Isolation Forest"] = preds == -1

                elif "DBSCAN" in method:
                    scaler_db = StandardScaler()
                    X_scaled = scaler_db.fit_transform(df_num)
                    db = DBSCAN(eps=0.5, min_samples=max(3, int(contamination * len(df_num))))
                    labels = db.fit_predict(X_scaled)
                    all_outlier_masks["DBSCAN"] = labels == -1

                elif "OPTICS" in method:
                    scaler_op = StandardScaler()
                    X_scaled = scaler_op.fit_transform(df_num)
                    op = OPTICS(min_samples=max(3, int(contamination * len(df_num))))
                    labels = op.fit_predict(X_scaled)
                    all_outlier_masks["OPTICS"] = labels == -1

            st.session_state.outlier_results = all_outlier_masks
            combined_mask = np.zeros(len(df_num), dtype=bool)
            for m in all_outlier_masks.values():
                combined_mask |= m
            st.session_state.outlier_mask = combined_mask
            st.session_state.outlier_indices = df_num.index[combined_mask]

            # Show results
            st.markdown(f"""
            <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:700;color:{CHARCOAL};margin:16px 0 12px">
                Detection Results
            </div>""", unsafe_allow_html=True)

            cols_r = st.columns(len(all_outlier_masks))
            for col, (name, mask) in zip(cols_r, all_outlier_masks.items()):
                n_out = mask.sum()
                pct = n_out / len(mask) * 100
                with col:
                    st.markdown(f"""
                    <div class="outlier-method">
                        <div class="outlier-method-name">{name}</div>
                        <div class="outlier-count">{n_out:,}</div>
                        <div class="outlier-sub">{pct:.1f}% of data · outliers detected</div>
                    </div>""", unsafe_allow_html=True)

            n_combined = combined_mask.sum()
            st.markdown(result_box(
                f"📊 Combined: {n_combined:,} outliers detected ({n_combined/len(df_num)*100:.1f}%)",
                f"Union of all selected methods. {len(df_work) - n_combined:,} clean samples remain.",
                "warning" if n_combined > 0 else "success"
            ), unsafe_allow_html=True)

            if PLOTLY_OK and len(num_cols) >= 2:
                fig = px.scatter(x=df_num[num_cols[0]], y=df_num[num_cols[1]],
                    color=np.where(combined_mask, "Outlier", "Normal"),
                    color_discrete_map={"Outlier": ANGRY_RED, "Normal": SAGE},
                    title=f"Outlier Scatter — {num_cols[0]} vs {num_cols[1]}")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Mono", size=10), height=360)
                st.plotly_chart(fig, use_container_width=True)

        # Remove outliers button
        if "outlier_mask" in st.session_state and st.session_state.outlier_mask is not None:
            n_out = st.session_state.outlier_mask.sum()
            if n_out > 0:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                c_rem1, c_rem2 = st.columns(2)
                with c_rem1:
                    if st.button(f"🗑️ Remove {n_out:,} outliers from dataset", key="remove_outliers"):
                        df_clean = df_work.copy()
                        outlier_idx = st.session_state.outlier_indices
                        df_clean = df_clean.drop(index=outlier_idx, errors="ignore").reset_index(drop=True)
                        st.session_state.df_cleaned = df_clean
                        st.session_state.outlier_mask = None
                        st.success(f"✅ Removed {n_out:,} outliers. Dataset now has {len(df_clean):,} rows.")
                        st.rerun()
                with c_rem2:
                    if st.button("✅ Keep outliers, proceed", key="keep_outliers"):
                        st.session_state.outlier_mask = None

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    nav_buttons(next_label="Next: Select Features →")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 4: FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_step_4():
    step_header("✂️", 5, "Feature Selection", "Use statistical methods to select the most informative features: variance threshold, correlation filter, and information gain.")

    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    if df is None or not st.session_state.feature_cols or not st.session_state.target_col:
        st.warning("Please complete the Data Input step first.")
        nav_buttons()
        return

    target = st.session_state.target_col
    features = st.session_state.feature_cols

    df_feat = df[features + [target]].dropna().copy()
    X = df_feat[features].copy()
    y = df_feat[target].copy()

    # Encode
    for col in X.select_dtypes(include=["object","category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype in ["object", "category"]:
        y = LabelEncoder().fit_transform(y.astype(str))
    y = pd.Series(y)

    num_features = X.select_dtypes(include=np.number).columns.tolist()

    method_tab1, method_tab2, method_tab3, method_tab4 = st.tabs([
        "📊 Variance Threshold", "🔗 Correlation Filter", "🎯 Info Gain / Mutual Info", "✅ Final Selection"
    ])

    with method_tab1:
        if not SKLEARN_OK:
            st.info("sklearn not available.")
        else:
            threshold = st.slider("Variance threshold (features below this are dropped)", 0.0, 1.0, 0.01, 0.01, key="var_thresh")
            sel = VarianceThreshold(threshold=threshold)
            try:
                sel.fit(X[num_features])
                kept = [f for f, s in zip(num_features, sel.get_support()) if s]
                dropped = [f for f, s in zip(num_features, sel.get_support()) if not s]
                variances = dict(zip(num_features, X[num_features].var().values))

                st.markdown(result_box(
                    f"✓ {len(kept)} of {len(num_features)} features pass the threshold",
                    f"Dropped {len(dropped)} low-variance features: {', '.join(dropped) if dropped else 'none'}",
                    "success" if dropped else "default"
                ), unsafe_allow_html=True)

                if PLOTLY_OK:
                    sorted_vars = sorted(variances.items(), key=lambda x: x[1], reverse=True)
                    fig = go.Figure(go.Bar(
                        x=[v for _, v in sorted_vars],
                        y=[k for k, _ in sorted_vars],
                        orientation="h",
                        marker_color=[SAGE if v > threshold else ANGRY_RED for k, v in sorted_vars]
                    ))
                    fig.add_vline(x=threshold, line_dash="dash", line_color=GOLD,
                        annotation_text=f"Threshold: {threshold}")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="DM Mono", size=10), height=max(300, len(num_features)*22),
                        title="Feature Variance", margin=dict(l=140, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                if st.button("Apply variance threshold selection", key="apply_var"):
                    st.session_state.selected_features = kept if kept else features
                    st.success(f"✅ Applied: {len(kept)} features selected")
            except Exception as e:
                st.warning(f"Variance threshold error: {e}")

    with method_tab2:
        corr_thresh = st.slider("Max absolute correlation allowed between features", 0.5, 1.0, 0.9, 0.05, key="corr_thresh")
        if len(num_features) >= 2:
            corr_matrix = X[num_features].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [c for c in upper.columns if any(upper[c] > corr_thresh)]
            kept_corr = [f for f in num_features if f not in to_drop]

            st.markdown(result_box(
                f"✓ {len(kept_corr)} features kept (dropped {len(to_drop)} highly correlated)",
                f"Removed: {', '.join(to_drop) if to_drop else 'none'} (correlation > {corr_thresh})",
                "success"
            ), unsafe_allow_html=True)

            if PLOTLY_OK and len(num_features) <= 20:
                fig = px.imshow(X[num_features].corr(),
                    color_continuous_scale=[[0, CHARCOAL],[0.5, CREAM_DK],[1, BURGUNDY]],
                    title="Feature Correlation Matrix", text_auto=".2f", aspect="auto")
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Mono", size=9), height=400)
                st.plotly_chart(fig, use_container_width=True)

            if st.button("Apply correlation filter", key="apply_corr"):
                st.session_state.selected_features = kept_corr if kept_corr else features
                st.success(f"✅ Applied: {len(kept_corr)} features selected")

    with method_tab3:
        if not SKLEARN_OK:
            st.info("sklearn not available. Install scikit-learn to use this feature.")
        elif not num_features:
            st.info("No numeric features available for mutual information analysis.")
        else:
            try:
                is_regression = st.session_state.problem_type == "Regression"
                # Ensure y is numeric for mutual_info
                y_mi = y.copy()
                if hasattr(y_mi, 'dtype') and y_mi.dtype in ['object', 'category']:
                    y_mi = LabelEncoder().fit_transform(y_mi.astype(str))
                mi_func = mutual_info_regression if is_regression else mutual_info_classif
                mi_scores = mi_func(X[num_features], y_mi, random_state=42)
                mi_df = pd.DataFrame({"Feature": num_features, "MI Score": mi_scores}).sort_values("MI Score", ascending=True)

                top_k = st.slider("Select top K features by mutual info", 1, len(num_features), min(10, len(num_features)), key="top_k_mi")
                top_features = mi_df.nlargest(top_k, "MI Score")["Feature"].tolist()

                if PLOTLY_OK:
                    colors_mi = [SAGE if f in top_features else CREAM_DK for f in mi_df["Feature"]]
                    fig = go.Figure(go.Bar(
                        x=mi_df["MI Score"], y=mi_df["Feature"],
                        orientation="h", marker_color=colors_mi
                    ))
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="DM Mono", size=10),
                        height=max(300, len(num_features)*24),
                        title=f"Mutual Information with Target ({target})",
                        margin=dict(l=140, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown(result_box(
                    f"Top {top_k} features by information gain",
                    ", ".join(top_features), "success"
                ), unsafe_allow_html=True)

                if st.button(f"Apply: keep top {top_k} features", key="apply_mi"):
                    st.session_state.selected_features = top_features
                    st.success(f"✅ Applied: {top_k} features selected by mutual info")
            except Exception as e:
                st.warning(f"Mutual info error: {e}")

    with method_tab4:
        current_sel = st.session_state.selected_features if st.session_state.selected_features else features
        final_sel = st.multiselect(
            "Manually adjust final feature selection",
            features, default=[f for f in current_sel if f in features], key="final_feat_sel"
        )
        if st.button("✅ Confirm feature selection", key="confirm_features"):
            st.session_state.selected_features = final_sel if final_sel else features
            st.success(f"✅ {len(st.session_state.selected_features)} features confirmed for modelling")

        if st.session_state.selected_features:
            st.markdown(result_box(
                f"✓ {len(st.session_state.selected_features)} features selected",
                ", ".join(st.session_state.selected_features[:15]) + ("..." if len(st.session_state.selected_features) > 15 else ""),
                "success"
            ), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    nav_buttons(next_label="Next: Split Data →")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 5: DATA SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_step_5():
    step_header("✂️", 6, "Data Split", "Divide your data into training and testing sets. Control the split ratio and optionally stratify by class.")

    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df
    target = st.session_state.target_col
    features = st.session_state.selected_features or st.session_state.feature_cols

    if df is None or not target or not features:
        st.warning("Complete previous steps first.")
        nav_buttons()
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20, 5, key="test_size_slider") / 100
        st.session_state.test_size = test_size
    with col2:
        # Auto-detect whether stratification is safe
        _is_clf = st.session_state.problem_type == "Classification"
        _default_strat = _is_clf  # default ON for classification
        stratify = st.checkbox("Stratified split", value=_default_strat, key="stratify")
    with col3:
        random_state = st.number_input("Random seed", 0, 999, 42, key="random_seed")

    df_split = df[features + [target]].dropna().copy()
    X_all = df_split[features].copy()
    y_all = df_split[target].copy()

    # Encode
    le_dict = {}
    for col in X_all.select_dtypes(include=["object","category"]).columns:
        le = LabelEncoder()
        X_all[col] = le.fit_transform(X_all[col].astype(str))
        le_dict[col] = le

    y_encoded = y_all
    if y_all.dtype in ["object","category"]:
        le_target = LabelEncoder()
        y_encoded = pd.Series(le_target.fit_transform(y_all.astype(str)), name=target)
        le_dict["__target__"] = le_target

    st.session_state.label_encoders = le_dict

    n_total = len(X_all)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test

    # Warn about stratification issues for high-cardinality / continuous targets
    n_unique_target = y_encoded.nunique() if hasattr(y_encoded, 'nunique') else len(set(y_encoded))
    min_class_count = pd.Series(y_encoded).value_counts().min() if n_unique_target < 5000 else 1
    stratify_safe = (n_unique_target <= 50 and min_class_count >= 2)

    if stratify and not stratify_safe:
        st.markdown(result_box(
            "⚠️ Stratified split not possible for this target",
            f"Target '{target}' has {n_unique_target} unique values. "
            "Stratification requires each class to have ≥ 2 members. "
            "The split will proceed without stratification automatically.",
            "warning"
        ), unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (v, l) in zip([c1,c2,c3,c4],[
        (f"{n_total:,}", "Total Samples"),
        (f"{n_train:,}", "Training Samples"),
        (f"{n_test:,}", "Testing Samples"),
        (f"{(1-test_size)*100:.0f}/{test_size*100:.0f}", "Train/Test %"),
    ]):
        with col:
            st.markdown(data_stat_card(v, l), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    if st.button("✂️ Split Data", key="split_btn", use_container_width=True):
        try:
            # Determine if stratification is actually safe
            use_strat = (stratify and _is_clf and stratify_safe)
            strat_arg = y_encoded if use_strat else None

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_encoded, test_size=test_size,
                stratify=strat_arg, random_state=int(random_state)
            )

            if stratify and not use_strat:
                st.info("ℹ️ Stratification was skipped (target has too many unique values). Used random split instead.")

            # Scale
            scaler = StandardScaler()
            X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
            X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)

            st.session_state.X_train = X_tr_s
            st.session_state.X_test = X_te_s
            st.session_state.y_train = y_tr.reset_index(drop=True)
            st.session_state.y_test = y_te.reset_index(drop=True)
            st.session_state.scaler_fitted = scaler

            st.success(f"✅ Split complete: {len(X_tr):,} train / {len(X_te):,} test samples")

            if PLOTLY_OK:
                if st.session_state.problem_type == "Classification":
                    tr_dist = y_tr.value_counts().reset_index()
                    te_dist = y_te.value_counts().reset_index()
                    tr_dist.columns = ["Class", "Count"]
                    te_dist.columns = ["Class", "Count"]

                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Distribution", "Testing Distribution"))
                    fig.add_trace(go.Bar(x=tr_dist["Class"].astype(str), y=tr_dist["Count"],
                        marker_color=BURGUNDY, name="Train"), row=1, col=1)
                    fig.add_trace(go.Bar(x=te_dist["Class"].astype(str), y=te_dist["Count"],
                        marker_color=TERRACOTTA, name="Test"), row=1, col=2)
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="DM Mono", size=10), height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Split failed: {e}")

    if st.session_state.X_train is not None:
        st.markdown(result_box(
            f"✓ Data split complete",
            f"Train: {len(st.session_state.X_train):,} rows · Test: {len(st.session_state.X_test):,} rows · {len(features)} features",
            "success"
        ), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    nav_buttons(next_label="Next: Select Model →")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 6: MODEL SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_step_6():
    step_header("🤖", 7, "Model Selection", "Choose the algorithm best suited for your problem type. Configure key hyperparameters before training.")

    problem = st.session_state.problem_type or "Classification"

    if problem == "Classification":
        models = [
            ("SVM", "🤖", "Support Vector Machine", "Powerful for high-dimensional data. Uses kernel trick to handle non-linear boundaries. Excellent accuracy on medium datasets.", ["kernel"]),
            ("Random Forest", "🌲", "Random Forest", "Ensemble of decision trees. Handles non-linearity, robust to noise, provides feature importance. Great default choice.", []),
            ("KNN", "🔍", "K-Nearest Neighbours", "Non-parametric method based on distance. Simple but effective for low-dimensional data with clear clusters.", ["n_neighbors"]),
            ("Logistic Regression", "📈", "Logistic Regression", "Linear baseline model. Fast, interpretable, works well when classes are linearly separable.", ["C"]),
        ]
    else:
        models = [
            ("Linear Regression", "📈", "Linear Regression", "Classic baseline for continuous prediction. Assumes a linear relationship between features and target.", []),
            ("SVR", "🤖", "Support Vector Regression", "SVM adapted for regression. Uses kernel trick; robust to outliers. Ideal for non-linear data.", ["kernel"]),
            ("Random Forest Regressor", "🌲", "Random Forest Regressor", "Ensemble method for regression. Handles non-linearity and interactions. Provides feature importance.", []),
        ]

    cols = st.columns(len(models))
    for col, (mkey, icon, name, desc, params) in zip(cols, models):
        with col:
            selected = st.session_state.model_name == mkey
            st.markdown(f"""
            <div class="model-card {'selected' if selected else ''}">
                <span class="model-icon">{icon}</span>
                <div class="model-name">{name}</div>
                <div class="model-desc">{desc}</div>
                {''.join(f'<span class="model-badge">{p}</span>' for p in params)}
            </div>""", unsafe_allow_html=True)
            if st.button(f"Select {name}", key=f"model_{mkey}", use_container_width=True):
                st.session_state.model_name = mkey
                st.session_state.model_params = {}
                st.rerun()

    if st.session_state.model_name:
        st.markdown(f"""
        <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{CHARCOAL};margin:24px 0 14px">
            Configure: {st.session_state.model_name}
        </div>""", unsafe_allow_html=True)

        params = {}
        if st.session_state.model_name in ["SVM", "SVR"]:
            c1, c2, c3 = st.columns(3)
            with c1: params["kernel"] = st.selectbox("Kernel", ["rbf","linear","poly","sigmoid"], key="svm_kernel")
            with c2: params["C"] = st.select_slider("C (regularisation)", [0.01,0.1,1.0,5.0,10.0,100.0], value=1.0, key="svm_c")
            with c3: params["gamma"] = st.selectbox("Gamma", ["scale","auto"], key="svm_gamma")
        elif st.session_state.model_name in ["Random Forest", "Random Forest Regressor"]:
            c1, c2 = st.columns(2)
            with c1: params["n_estimators"] = st.select_slider("N estimators", [50,100,200,300], value=100, key="rf_ne")
            with c2: params["max_depth"] = st.select_slider("Max depth", [None,3,5,10,20], value=None, key="rf_md")
        elif st.session_state.model_name == "KNN":
            params["n_neighbors"] = st.slider("K (neighbors)", 1, 20, 5, key="knn_k")
        elif st.session_state.model_name == "Logistic Regression":
            params["C"] = st.select_slider("C (regularisation)", [0.01,0.1,1.0,10.0,100.0], value=1.0, key="lr_c")

        st.session_state.model_params = params

        st.markdown(result_box(
            f"✓ Model selected: {st.session_state.model_name}",
            f"Parameters: {params if params else 'defaults'}. Click Next to proceed to training.",
            "success"
        ), unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    nav_buttons(next_label="Next: Train Model →")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 7: TRAINING + KFOLD
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_step_7():
    step_header("🏋️", 8, "Model Training & K-Fold Validation", "Train your model with cross-validation to get a robust estimate of performance across multiple data splits.")

    if st.session_state.X_train is None or st.session_state.model_name is None:
        st.warning("Please complete the Data Split and Model Selection steps first.")
        nav_buttons()
        return

    X_tr = st.session_state.X_train
    y_tr = st.session_state.y_train
    X_te = st.session_state.X_test
    y_te = st.session_state.y_test
    params = st.session_state.model_params or {}
    is_clf = st.session_state.problem_type == "Classification"

    c1, c2 = st.columns(2)
    with c1:
        k = st.slider("Number of folds (K)", 2, 10, 5, key="kfold_k")
        st.session_state.k_folds = k
    with c2:
        shuffle = st.checkbox("Shuffle before splitting", value=True, key="kfold_shuffle")

    if st.button("🏋️ Train & Cross-Validate", key="train_btn", use_container_width=True):
        with st.spinner(f"Training {st.session_state.model_name} with {k}-fold CV…"):
            try:
                # Build model
                mname = st.session_state.model_name
                if mname == "SVM":
                    model = SVC(kernel=params.get("kernel","rbf"), C=params.get("C",1.0),
                        gamma=params.get("gamma","scale"), probability=True, random_state=42)
                elif mname == "SVR":
                    model = SVR(kernel=params.get("kernel","rbf"), C=params.get("C",1.0),
                        gamma=params.get("gamma","scale"))
                elif mname == "Random Forest":
                    model = RandomForestClassifier(n_estimators=params.get("n_estimators",100),
                        max_depth=params.get("max_depth",None), random_state=42, n_jobs=-1)
                elif mname == "Random Forest Regressor":
                    model = RandomForestRegressor(n_estimators=params.get("n_estimators",100),
                        max_depth=params.get("max_depth",None), random_state=42, n_jobs=-1)
                elif mname == "KNN":
                    model = KNeighborsClassifier(n_neighbors=params.get("n_neighbors",5))
                elif mname == "Logistic Regression":
                    model = LogisticRegression(C=params.get("C",1.0), random_state=42, max_iter=1000)
                else:
                    model = LinearRegression()

                # KFold CV
                scoring = "accuracy" if is_clf else "r2"
                rs = 42 if shuffle else None
                # For classification, try StratifiedKFold first, fall back to regular KFold
                if is_clf and shuffle:
                    try:
                        cv = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=rs)
                        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=scoring)
                    except ValueError:
                        # StratifiedKFold fails when classes have too few members
                        cv = KFold(n_splits=k, shuffle=shuffle, random_state=rs)
                        cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=scoring)
                        st.info("ℹ️ Switched from StratifiedKFold to regular KFold (target has too many unique classes).")
                else:
                    cv = KFold(n_splits=k, shuffle=shuffle, random_state=rs)
                    cv_scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=scoring)
                st.session_state.cv_scores = cv_scores

                # Final model fit
                model.fit(X_tr, y_tr)
                st.session_state.model = model

                if is_clf:
                    train_score = accuracy_score(y_tr, model.predict(X_tr))
                    test_score = accuracy_score(y_te, model.predict(X_te))
                else:
                    train_score = r2_score(y_tr, model.predict(X_tr))
                    test_score = r2_score(y_te, model.predict(X_te))

                st.session_state.train_score = train_score
                st.session_state.test_score = test_score

                st.success(f"✅ Training complete! CV mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.session_state.cv_scores is not None:
        scores = st.session_state.cv_scores
        scoring_label = "Accuracy" if is_clf else "R²"

        sc1, sc2, sc3, sc4 = st.columns(4)
        for col, (v, l) in zip([sc1,sc2,sc3,sc4],[
            (f"{scores.mean():.4f}", f"Mean CV {scoring_label}"),
            (f"{scores.std():.4f}", "Std Deviation"),
            (f"{scores.min():.4f}", "Min Score"),
            (f"{scores.max():.4f}", "Max Score"),
        ]):
            with col:
                st.markdown(data_stat_card(v, l), unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Fold bars
        st.markdown(f"""
        <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:700;color:{CHARCOAL};margin-bottom:14px">
            Per-Fold {scoring_label}
        </div>""", unsafe_allow_html=True)

        max_s = max(scores) if max(scores) > 0 else 1
        fold_html = ""
        for i, s in enumerate(scores):
            pct = (s / max_s) * 100 if max_s > 0 else 0
            color = SAGE if s >= scores.mean() else TERRACOTTA
            fold_html += f"""
            <div class="fold-bar-container">
                <span class="fold-label">Fold {i+1}</span>
                <div class="fold-track"><div class="fold-fill" style="width:{pct:.1f}%;background:{color}"></div></div>
                <span class="fold-val">{s:.4f}</span>
            </div>"""
        st.markdown(fold_html, unsafe_allow_html=True)

        if PLOTLY_OK:
            fig = go.Figure()
            bar_colors = [SAGE if s >= scores.mean() else TERRACOTTA for s in scores]
            fig.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(len(scores))], y=scores,
                marker_color=bar_colors, name="Fold Score"))
            fig.add_hline(y=scores.mean(), line_dash="dash", line_color=GOLD,
                annotation_text=f"Mean: {scores.mean():.4f}")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Mono", size=10), height=300,
                yaxis_title=scoring_label, title=f"{k}-Fold CV Results")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    nav_buttons(next_label="Next: View Metrics →")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 8: PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_step_8():
    step_header("📈", 9, "Performance Metrics", "Evaluate your trained model with detailed metrics. Check for overfitting and underfitting based on train/test gap.")

    if st.session_state.model is None or st.session_state.X_test is None:
        st.warning("Please complete the Training step first.")
        nav_buttons()
        return

    model = st.session_state.model
    X_te = st.session_state.X_test
    y_te = st.session_state.y_test
    X_tr = st.session_state.X_train
    y_tr = st.session_state.y_train
    is_clf = st.session_state.problem_type == "Classification"

    y_pred_te = model.predict(X_te)
    y_pred_tr = model.predict(X_tr)

    if is_clf:
        # Classification metrics
        train_acc = accuracy_score(y_tr, y_pred_tr)
        test_acc = accuracy_score(y_te, y_pred_te)
        prec = precision_score(y_te, y_pred_te, average="weighted", zero_division=0)
        rec = recall_score(y_te, y_pred_te, average="weighted", zero_division=0)
        f1 = f1_score(y_te, y_pred_te, average="weighted", zero_division=0)

        c1,c2,c3,c4,c5 = st.columns(5)
        for col, (v, l) in zip([c1,c2,c3,c4,c5],[
            (f"{test_acc*100:.2f}%", "Test Accuracy"),
            (f"{train_acc*100:.2f}%", "Train Accuracy"),
            (f"{prec*100:.2f}%", "Precision"),
            (f"{rec*100:.2f}%", "Recall"),
            (f"{f1*100:.2f}%", "F1 Score"),
        ]):
            with col:
                st.markdown(data_stat_card(v, l), unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Confusion matrix
        if PLOTLY_OK:
            cm = confusion_matrix(y_te, y_pred_te)
            labels = sorted(y_te.unique())
            fig = px.imshow(cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
                color_continuous_scale=[[0, CREAM],[0.5, TERRACOTTA],[1, BURGUNDY]],
                title="Confusion Matrix", text_auto=True, aspect="auto")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Mono", size=11), height=400)
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Regression metrics
        train_r2 = r2_score(y_tr, y_pred_tr)
        test_r2 = r2_score(y_te, y_pred_te)
        mae = mean_absolute_error(y_te, y_pred_te)
        mse = mean_squared_error(y_te, y_pred_te)
        rmse = np.sqrt(mse)

        c1,c2,c3,c4,c5 = st.columns(5)
        for col, (v, l) in zip([c1,c2,c3,c4,c5],[
            (f"{test_r2:.4f}", "Test R²"),
            (f"{train_r2:.4f}", "Train R²"),
            (f"{mae:.4f}", "MAE"),
            (f"{rmse:.4f}", "RMSE"),
            (f"{mse:.4f}", "MSE"),
        ]):
            with col:
                st.markdown(data_stat_card(v, l), unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        if PLOTLY_OK:
            fig = px.scatter(x=y_te, y=y_pred_te, color_discrete_sequence=[BURGUNDY],
                title="Actual vs Predicted", labels={"x":"Actual","y":"Predicted"})
            mn, mx = min(y_te.min(), y_pred_te.min()), max(y_te.max(), y_pred_te.max())
            fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                line=dict(color=GOLD, dash="dash", width=2))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Mono", size=10), height=380)
            st.plotly_chart(fig, use_container_width=True)

        train_r2 = train_r2
        test_r2 = test_r2

    # ── Overfitting / Underfitting Detection ───────────────────────────────────
    st.markdown(f"""
    <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{CHARCOAL};margin:24px 0 14px">
        Overfitting / Underfitting Analysis
    </div>""", unsafe_allow_html=True)

    if is_clf:
        tr_s, te_s = train_acc, test_acc
    else:
        tr_s, te_s = train_r2, test_r2

    gap = tr_s - te_s
    metric_name = "Accuracy" if is_clf else "R²"

    if gap > 0.15:
        kind, label, explanation, dot_color = (
            "danger", "⚠️ Overfitting Detected",
            f"Train {metric_name} ({tr_s*100:.1f}%) is much higher than Test ({te_s*100:.1f}%). "
            "Gap: {gap*100:.1f}%. Model has memorized training data. "
            "Try: reduce model complexity, add regularisation, or collect more data.",
            ANGRY_RED
        )
    elif te_s < 0.5 and tr_s < 0.5:
        kind, label, explanation, dot_color = (
            "warning", "📉 Underfitting Detected",
            f"Both Train ({tr_s*100:.1f}%) and Test ({te_s*100:.1f}%) scores are low. "
            "Model is too simple. Try: increase complexity, add features, or try a more powerful algorithm.",
            AMBER
        )
    else:
        kind, label, explanation, dot_color = (
            "success", "✅ Good Fit",
            f"Train {metric_name}: {tr_s*100:.1f}% · Test {metric_name}: {te_s*100:.1f}% · Gap: {gap*100:.1f}%. "
            "Model generalises well to unseen data.",
            SAGE
        )

    st.markdown(f"""
    <div class="overfit-indicator" style="background:{'rgba(58,107,62,0.08)' if kind=='success' else 'rgba(139,45,26,0.08)' if kind=='danger' else 'rgba(224,139,32,0.08)'}">
        <div class="overfit-dot" style="background:{dot_color}"></div>
        <div>
            <div class="overfit-text">{label}</div>
            <div class="overfit-sub">{explanation}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if PLOTLY_OK:
        fig = go.Figure(go.Bar(
            x=["Training", "Test"],
            y=[tr_s * 100, te_s * 100],
            marker_color=[TERRACOTTA, BURGUNDY],
            text=[f"{tr_s*100:.2f}%", f"{te_s*100:.2f}%"],
            textposition="outside"
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Mono", size=11), height=300,
            yaxis_title=f"{metric_name} (%)", title="Train vs Test Performance")
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance (tree models)
    if hasattr(model, "feature_importances_") and PLOTLY_OK:
        st.markdown(f"""
        <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:{CHARCOAL};margin:20px 0 14px">
            Feature Importances
        </div>""", unsafe_allow_html=True)
        feat_names = st.session_state.X_train.columns.tolist()
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=True)
        fig_fi = go.Figure(go.Bar(x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h", marker_color=TERRACOTTA))
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Mono", size=10),
            height=max(300, len(feat_names)*22),
            title="Feature Importances", margin=dict(l=140, r=20, t=40, b=20))
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    nav_buttons(next_label="Next: Tune Hyperparameters →")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 9: HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
def pipeline_step_9():
    step_header("⚙️", 10, "Hyperparameter Tuning", "Optimise your model's hyperparameters using GridSearchCV or RandomizedSearchCV to push performance further.")

    if st.session_state.model is None:
        st.warning("Please complete the Training step first.")
        nav_buttons(next_label="Finish ✓")
        return

    model = st.session_state.model
    X_tr = st.session_state.X_train
    y_tr = st.session_state.y_train
    X_te = st.session_state.X_test
    y_te = st.session_state.y_test
    mname = st.session_state.model_name
    is_clf = st.session_state.problem_type == "Classification"

    scoring = "accuracy" if is_clf else "r2"

    # Define param grids per model
    param_grids = {
        "SVM": {"C": [0.1, 1, 5, 10, 50], "kernel": ["rbf","linear","poly"], "gamma": ["scale","auto"]},
        "SVR": {"C": [0.1, 1, 5, 10], "kernel": ["rbf","linear"], "gamma": ["scale","auto"]},
        "Random Forest": {"n_estimators": [50,100,200], "max_depth": [None,5,10,20], "min_samples_split": [2,5,10]},
        "Random Forest Regressor": {"n_estimators": [50,100,200], "max_depth": [None,5,10,20], "min_samples_split": [2,5,10]},
        "KNN": {"n_neighbors": [3,5,7,9,11,15], "weights": ["uniform","distance"], "metric": ["euclidean","manhattan"]},
        "Logistic Regression": {"C": [0.01,0.1,1,10,100], "solver": ["lbfgs","saga"]},
        "Linear Regression": {},
    }

    grid = param_grids.get(mname, {})

    col1, col2 = st.columns(2)
    with col1:
        search_method = st.selectbox("Search strategy", ["GridSearchCV","RandomizedSearchCV"], key="tune_method")
    with col2:
        cv_k = st.slider("CV folds for tuning", 2, 10, 5, key="tune_cv")

    if search_method == "RandomizedSearchCV":
        n_iter = st.slider("Number of random iterations", 5, 50, 20, key="tune_n_iter")

    if not grid:
        st.info("No hyperparameters to tune for this model.")
        nav_buttons(next_label="Finish ✓")
        return

    st.markdown(f"""
    <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:700;color:{CHARCOAL};margin:16px 0 10px">
        Search Space for {mname}
    </div>""", unsafe_allow_html=True)

    grid_html = ""
    for param, values in grid.items():
        values_str = ", ".join(str(v) for v in values)
        grid_html += f"""
        <div style="display:flex;gap:12px;padding:10px 14px;background:{PARCHMENT};
            border:1px solid {CREAM_DK};border-radius:8px;margin-bottom:8px;align-items:center">
            <span style="font-family:'DM Mono',monospace;font-size:0.8rem;font-weight:700;color:{BURGUNDY};min-width:130px">{param}</span>
            <span style="font-size:0.8rem;color:{SLATE}">[{values_str}]</span>
            <span style="margin-left:auto;font-family:'DM Mono',monospace;font-size:0.72rem;color:{MUTED}">{len(values)} options</span>
        </div>"""
    st.markdown(grid_html, unsafe_allow_html=True)

    # Before score
    if is_clf:
        before_score = accuracy_score(y_te, model.predict(X_te))
    else:
        before_score = r2_score(y_te, model.predict(X_te))

    if st.button(f"⚙️ Run {search_method}", key="tune_btn", use_container_width=True):
        with st.spinner(f"Running {search_method}… this may take a moment"):
            try:
                # Rebuild base estimator
                if mname == "SVM":
                    base = SVC(probability=True, random_state=42)
                elif mname == "SVR":
                    base = SVR()
                elif mname == "Random Forest":
                    base = RandomForestClassifier(random_state=42, n_jobs=-1)
                elif mname == "Random Forest Regressor":
                    base = RandomForestRegressor(random_state=42, n_jobs=-1)
                elif mname == "KNN":
                    base = KNeighborsClassifier()
                elif mname == "Logistic Regression":
                    base = LogisticRegression(max_iter=1000, random_state=42)
                else:
                    base = LinearRegression()

                if search_method == "GridSearchCV":
                    searcher = GridSearchCV(base, grid, cv=cv_k, scoring=scoring,
                        n_jobs=-1, verbose=0, refit=True)
                else:
                    searcher = RandomizedSearchCV(base, grid, n_iter=n_iter, cv=cv_k,
                        scoring=scoring, n_jobs=-1, verbose=0, refit=True, random_state=42)

                searcher.fit(X_tr, y_tr)
                best_model = searcher.best_estimator_

                if is_clf:
                    after_score = accuracy_score(y_te, best_model.predict(X_te))
                else:
                    after_score = r2_score(y_te, best_model.predict(X_te))

                st.session_state.tuned_model = best_model
                st.session_state.tuning_results = {
                    "best_params": searcher.best_params_,
                    "best_cv_score": searcher.best_score_,
                    "before_score": before_score,
                    "after_score": after_score,
                    "results_df": pd.DataFrame(searcher.cv_results_).sort_values("rank_test_score").head(10),
                }
                st.success(f"✅ Tuning complete! Best CV score: {searcher.best_score_:.4f}")

            except Exception as e:
                st.error(f"Tuning failed: {e}")

    if st.session_state.tuning_results:
        res = st.session_state.tuning_results
        metric = "Accuracy" if is_clf else "R²"
        before = res["before_score"]
        after = res["after_score"]
        improvement = (after - before) * 100

        c1, c2, c3, c4 = st.columns(4)
        for col, (v, l) in zip([c1,c2,c3,c4],[
            (f"{before*100:.2f}%", f"Before ({metric})"),
            (f"{after*100:.2f}%", f"After ({metric})"),
            (f"{'+' if improvement >= 0 else ''}{improvement:.2f}%", "Improvement"),
            (f"{res['best_cv_score']*100:.2f}%", "Best CV Score"),
        ]):
            with col:
                st.markdown(data_stat_card(v, l), unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        kind = "success" if improvement > 0.1 else ("warning" if abs(improvement) < 0.1 else "danger")
        st.markdown(result_box(
            f"{'✅ Performance improved' if improvement > 0 else '🔄 No significant change'}",
            f"Best parameters: {res['best_params']}",
            kind
        ), unsafe_allow_html=True)

        if PLOTLY_OK:
            fig = go.Figure(go.Bar(
                x=["Before Tuning", "After Tuning"],
                y=[before * 100, after * 100],
                marker_color=[SLATE, BURGUNDY],
                text=[f"{before*100:.2f}%", f"{after*100:.2f}%"],
                textposition="outside"
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Mono", size=11), height=300,
                title=f"Hyperparameter Tuning Impact — {metric}", yaxis_title=f"{metric} (%)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div style="font-family:'Playfair Display',serif;font-size:1rem;font-weight:700;color:{CHARCOAL};margin:20px 0 10px">
            Top 10 Parameter Combinations
        </div>""", unsafe_allow_html=True)

        display_df = res["results_df"][["params","mean_test_score","std_test_score","rank_test_score"]].copy()
        display_df.columns = ["Parameters", f"Mean CV {metric}", "Std", "Rank"]
        display_df[f"Mean CV {metric}"] = display_df[f"Mean CV {metric}"].map(lambda x: f"{x:.4f}")
        display_df["Std"] = display_df["Std"].map(lambda x: f"{x:.4f}")
        st.dataframe(display_df, use_container_width=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{CHARCOAL};border-radius:18px;padding:28px 24px;text-align:center">
            <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;color:{GOLD};margin-bottom:10px">
                🎉 ML Pipeline Complete!
            </div>
            <p style="color:rgba(247,242,232,0.6);font-size:0.9rem;margin-bottom:16px">
                You've completed all 9 steps of the ML pipeline — from problem framing to tuned model deployment.
            </p>
            {''.join(f'<span style="display:inline-block;padding:3px 10px;margin:3px;background:rgba(212,175,55,0.1);border:1px solid rgba(212,175,55,0.2);border-radius:12px;font-size:0.72rem;font-family:DM Mono,monospace;color:{GOLD}">{s}</span>' for s in ["Problem Defined","Data Loaded","EDA Done","Data Cleaned","Features Selected","Data Split","Model Trained","CV Validated","Metrics Checked","Hyperparams Tuned"])}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    nav_buttons(next_label="🏠 Back to Home →")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ML PIPELINE LAB
# ─────────────────────────────────────────────────────────────────────────────
def page_pipeline():
    step = st.session_state.pipeline_step
    st.markdown(render_stepper(step), unsafe_allow_html=True)
    st.markdown('<div class="pipeline-content">', unsafe_allow_html=True)

    step_fns = [
        pipeline_step_0,
        pipeline_step_1,
        pipeline_step_2,
        pipeline_step_3,
        pipeline_step_4,
        pipeline_step_5,
        pipeline_step_6,
        pipeline_step_7,
        pipeline_step_8,
        pipeline_step_9,
    ]

    if step < len(step_fns):
        step_fns[step]()
    else:
        page_home()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
def render_footer():
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="mf-footer">
        <span class="mf-ft-brand">🎬 MoodFlix ✦ ML Pipeline Lab</span>
        <span class="mf-ft-txt">SVM · K-Means · PCA · TMDb · scikit-learn · Plotly</span>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
p = st.session_state.page
if   p == "home":     page_home()
elif p == "rec":      page_rec()
elif p == "met":      page_metrics()
elif p == "abt":      page_about()
elif p == "pipeline": page_pipeline()
else:                 page_home()

render_footer()
