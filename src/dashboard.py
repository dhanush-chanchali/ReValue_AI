"""
ReValue AI — Ultra-Premium Dark Electronics Resale & Appraisal Platform
========================================================================
- Luxury Dark Obsidian & Champagne Gold + Electric Emerald design theme
- Interactive dynamic glowing cursor with magnetic trail & particle sparkles
- Expanded 120+ flagship & popular smartphone catalogue across 10 top brands
- 100% Indian Rupees (₹ / INR) native — zero dollars ($)
- Official 7-stage diagnostic appraisal questionnaire
- Fixed log_prediction argument alignment in M1-M6 monitoring
- Zero third-party brand references (Proprietary ReValue AI brand throughout)
"""

import os, sys, json, time, joblib, warnings, math
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monitoring import (
    get_recent_logs, get_all_logs_df, get_log_count,
    detect_data_drift, get_model_health_summary,
    compute_latency_stats, log_prediction,
    SLA_LATENCY_P95_MS, SLA_ANOMALY_RATE_PCT
)
from src.retrain import trigger_retraining
from src.wizards import render_phone_wizard, render_laptop_wizard
from src.syllabus_view import render_syllabus_page

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DATA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
USD_TO_INR = 84.0

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReValue AI Pro — Elite Device Resale & Instant Trade-In",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "ReValue AI — ML-Powered Luxury Electronics Valuation Platform"}
)

# ─────────────────────────────────────────────────────────────────────────────
# ULTRA-PREMIUM LUXURY CSS DESIGN SYSTEM (GOLD & OBSIDIAN EMERALD)
# ─────────────────────────────────────────────────────────────────────────────
LUXURY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=Cinzel:wght@600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-base:        #040711;
    --bg-surface:     #080d1a;
    --bg-card:        #0d1424;
    --bg-card-alt:    #121b30;
    --bg-card-hover:  #16223d;
    --bdr-subtle:     rgba(255, 255, 255, 0.06);
    --bdr-card:       rgba(212, 175, 55, 0.18);
    --bdr-card-hover: rgba(0, 229, 153, 0.45);
    --bdr-glow:       rgba(212, 175, 55, 0.45);
    
    /* Luxury Color Tokens */
    --gold:           #d4af37;
    --gold-light:     #f6e05e;
    --gold-dark:      #997a15;
    --emerald:        #00e599;
    --emerald-dark:   #00b377;
    --cyan:           #06b6d4;
    --amber:          #f59e0b;
    --red:            #ef4444;
    
    --text-white:     #ffffff;
    --text-gold:      #fce7a1;
    --text-1:         #f1f5f9;
    --text-2:         #94a3b8;
    --text-3:         #64748b;
    --font:           'Plus Jakarta Sans', -apple-system, sans-serif;
    --font-heading:   'Cinzel', serif;
    --mono:           'JetBrains Mono', monospace;
}

*, *::before, *::after {
    box-sizing: border-box;
}

/* Force dark theme throughout entire Streamlit frame with luxury ambient glow */
html, body, .stApp, 
[data-testid="stAppViewContainer"], 
[data-testid="stHeader"], 
[data-testid="stToolbar"], 
section.main, 
.block-container {
    background-color: var(--bg-base) !important;
    background: radial-gradient(circle at 85% 15%, rgba(212, 175, 55, 0.05) 0%, transparent 45%),
                radial-gradient(circle at 15% 20%, rgba(0, 229, 153, 0.06) 0%, transparent 45%),
                radial-gradient(circle at 50% 85%, rgba(6, 182, 212, 0.04) 0%, transparent 55%),
                var(--bg-base) !important;
    color: var(--text-1) !important;
    font-family: var(--font) !important;
    -webkit-font-smoothing: antialiased;
}

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1260px !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}
[data-testid="stSidebar"] {
    display: none !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}

/* LUXURY NAVBAR */
.revalue-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(13, 20, 36, 0.88);
    backdrop-filter: blur(24px);
    border: 1px solid var(--bdr-card);
    border-radius: 22px;
    padding: 16px 32px;
    margin-bottom: 24px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
}
.revalue-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 1.45rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    color: var(--text-white);
}
.revalue-logo-badge {
    background: linear-gradient(135deg, #d4af37, #00e599);
    color: #040711;
    font-size: 0.68rem;
    font-weight: 900;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(212, 175, 55, 0.3);
}
.revalue-trust-tag {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(212, 175, 55, 0.08);
    border: 1px solid rgba(212, 175, 55, 0.25);
    padding: 7px 16px;
    border-radius: 30px;
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--gold-light);
}

/* HERO SECTION */
.hero-card {
    background: linear-gradient(145deg, rgba(18, 27, 48, 0.95), rgba(8, 13, 26, 0.95));
    border: 1.5px solid var(--bdr-card);
    border-radius: 30px;
    padding: 48px 52px;
    position: relative;
    overflow: hidden;
    margin-bottom: 32px;
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(212, 175, 55, 0.25);
}
.hero-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 15%;
    right: 15%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), var(--emerald), transparent);
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(212, 175, 55, 0.12);
    border: 1px solid rgba(212, 175, 55, 0.35);
    color: var(--gold-light);
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 100px;
    margin-bottom: 20px;
}
.hero-badge-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--gold);
    box-shadow: 0 0 12px var(--gold);
}
.hero-title {
    font-size: 3.4rem;
    font-weight: 900;
    color: var(--text-white);
    line-height: 1.1;
    letter-spacing: -0.04em;
    margin: 0 0 16px 0;
}
.hero-title span {
    background: linear-gradient(135deg, #f6e05e 0%, #d4af37 40%, #00e599 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.08rem;
    color: var(--text-2);
    line-height: 1.65;
    max-width: 620px;
    margin-bottom: 26px;
}
.hero-perks {
    display: flex;
    flex-wrap: wrap;
    gap: 22px;
    margin-top: 10px;
}
.hero-perk-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--text-1);
}
.hero-perk-icon {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: rgba(212, 175, 55, 0.2);
    color: var(--gold-light);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
}

/* TRUST BANNER */
.revalue-perks-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
    margin-bottom: 32px;
}
.revalue-perk-card {
    background: var(--bg-card);
    border: 1px solid var(--bdr-card);
    border-radius: 20px;
    padding: 24px 20px;
    text-align: center;
    transition: all 0.28s ease;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}
.revalue-perk-card:hover {
    border-color: var(--emerald);
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 229, 153, 0.15);
}
.revalue-perk-icon {
    font-size: 2rem;
    margin-bottom: 12px;
    display: inline-block;
}
.revalue-perk-title {
    font-size: 0.96rem;
    font-weight: 800;
    color: var(--text-white);
    margin-bottom: 6px;
}
.revalue-perk-desc {
    font-size: 0.78rem;
    color: var(--text-2);
    line-height: 1.5;
}

/* QUESTIONNAIRE CARDS */
.q-card {
    background: var(--bg-card);
    border: 1px solid var(--bdr-card);
    border-radius: 24px;
    padding: 30px;
    margin-bottom: 24px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.35);
    position: relative;
    transition: border-color 0.25s ease;
}
.q-card:hover {
    border-color: rgba(212, 175, 55, 0.4);
}
.q-title {
    font-size: 1.2rem;
    font-weight: 800;
    color: var(--text-white);
    margin: 0 0 6px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.q-sub {
    font-size: 0.85rem;
    color: var(--text-2);
    margin-bottom: 22px;
    line-height: 1.55;
}

/* FINAL VALUATION HERO */
.valuation-hero {
    background: radial-gradient(circle at 85% 15%, rgba(212, 175, 55, 0.15) 0%, transparent 60%),
                radial-gradient(circle at 20% 85%, rgba(0, 229, 153, 0.12) 0%, transparent 60%),
                linear-gradient(145deg, #0f182e 0%, #162442 100%);
    border: 1.8px solid var(--gold);
    border-radius: 30px;
    padding: 48px 52px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 24px 70px rgba(0, 0, 0, 0.65), 0 0 40px rgba(212, 175, 55, 0.15);
    margin-bottom: 32px;
}
.valuation-hero::before {
    content: '';
    position: absolute;
    top: 0;
    left: 15%;
    right: 15%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), var(--emerald), transparent);
}
.valuation-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.74rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--gold-light);
    margin-bottom: 16px;
}
.valuation-amount {
    font-size: 4.8rem;
    font-weight: 900;
    color: var(--text-white);
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 14px;
}
.valuation-amount span {
    color: var(--gold);
    font-size: 3rem;
    font-weight: 700;
    margin-right: 4px;
}
.valuation-range-pill {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    background: rgba(212, 175, 55, 0.12);
    border: 1px solid rgba(212, 175, 55, 0.35);
    border-radius: 100px;
    padding: 8px 22px;
    font-size: 0.88rem;
    color: var(--text-1);
    margin-bottom: 26px;
}
.valuation-breakdown-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 13px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.valuation-breakdown-row:last-child {
    border-bottom: none;
}
.val-label {
    font-size: 0.9rem;
    color: var(--text-2);
}
.val-val {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-1);
    font-family: var(--mono);
}
.val-val.plus {
    color: var(--emerald);
}
.val-val.minus {
    color: var(--red);
}

/* STREAMLIT NATIVE CONTROLS LUXURY OVERRIDES */
.stButton > button {
    background: linear-gradient(135deg, #d4af37 0%, #e5c158 20%, #00e599 80%, #00b377 100%) !important;
    color: #040711 !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 900 !important;
    font-size: 0.94rem !important;
    padding: 0.75rem 1.8rem !important;
    box-shadow: 0 6px 24px rgba(212, 175, 55, 0.35) !important;
    transition: all 0.22s ease !important;
    font-family: var(--font) !important;
    letter-spacing: -0.01em !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px rgba(0, 229, 153, 0.5) !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255, 255, 255, 0.05) !important;
    color: var(--text-1) !important;
    border: 1px solid var(--bdr-card) !important;
    box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(212, 175, 55, 0.12) !important;
    border-color: var(--gold) !important;
    color: var(--gold-light) !important;
}

/* Selectboxes, inputs, sliders */
.stSelectbox > div > div {
    background: var(--bg-surface) !important;
    border: 1px solid var(--bdr-card) !important;
    border-radius: 14px !important;
    color: var(--text-white) !important;
}
.stSelectbox label, .stSlider label, .stNumberInput label, .stTextInput label, .stRadio label, .stCheckbox label {
    color: var(--text-1) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: -0.01em !important;
}
.stNumberInput > div > div > input, .stTextInput > div > div > input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--bdr-card) !important;
    border-radius: 14px !important;
    color: var(--text-white) !important;
    font-size: 0.96rem !important;
    padding: 11px 16px !important;
}
/* High contrast custom slider styling - zero green background on numbers */
div[data-testid="stSlider"] div[data-baseweb="slider"] {
    background: transparent !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] div {
    background: #1e293b !important;
}
div[data-testid="stSlider"] div[role="slider"] {
    background: #ffffff !important;
    border: 2px solid #38bdf8 !important;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.8) !important;
}
div[data-testid="stSlider"] div[data-testid="stThumbValue"],
div[data-testid="stSlider"] div[aria-hidden="true"] {
    background: #080d1a !important;
    color: #ffffff !important;
    border: 1.5px solid #38bdf8 !important;
    border-radius: 8px !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    padding: 3px 8px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.85) !important;
}
div[data-testid="stSlider"] span {
    color: #cbd5e1 !important;
    font-weight: 700 !important;
}
.stRadio > div {
    background: transparent !important;
    gap: 12px !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 10px !important;
    background: var(--bg-surface) !important;
    padding: 8px !important;
    border-radius: 16px !important;
    border: 1px solid var(--bdr-card) !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-2) !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 9px 20px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--gold-light) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4), inset 0 0 0 1px rgba(212, 175, 55, 0.3) !important;
}
div[data-testid="stDataFrame"] {
    border-radius: 18px !important;
    border: 1px solid var(--bdr-card) !important;
    overflow: hidden !important;
}

/* Metric card */
.kpi-tile {
    background: var(--bg-card);
    border: 1px solid var(--bdr-card);
    border-radius: 18px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
}
.kpi-tile::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), var(--emerald), transparent);
}
.kpi-lbl {
    font-size: 0.74rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--gold-light);
    margin-bottom: 8px;
}
.kpi-val {
    font-size: 1.95rem;
    font-weight: 900;
    color: var(--emerald);
    letter-spacing: -0.03em;
    font-family: var(--mono);
}
.kpi-sub {
    font-size: 0.78rem;
    color: var(--text-2);
    margin-top: 6px;
}
/* ── Premium Crisp KaTeX LaTeX Mathematical Equations Display ── */
div[data-testid="stLatex"] {
    background: #080d1a !important;
    border-left: 1.5px solid #38bdf8 !important;
    border-right: 1.5px solid #38bdf8 !important;
    border-top: none !important;
    border-bottom: none !important;
    padding: 16px 24px !important;
    margin: 0 !important;
    overflow-x: auto !important;
    display: flex !important;
    justify-content: center !important;
}
div[data-testid="stLatex"] .katex-display {
    margin: 4px 0 !important;
}
div[data-testid="stLatex"] .katex {
    color: #f8fafc !important;
    font-size: 1.35rem !important;
    text-shadow: 0 0 16px rgba(56, 189, 248, 0.3) !important;
}
</style>
"""
st.markdown(LUXURY_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE ANIMATED CURSOR TRAIL & SPARKLE PARTICLES
# ─────────────────────────────────────────────────────────────────────────────
CURSOR_JS = """
<script>
(function() {
    try {
        const doc = window.parent.document;
        if (!doc) return;

        let dot = doc.getElementById('revalue-cursor-dot');
        let ring = doc.getElementById('revalue-cursor-ring');

        if (!dot) {
            dot = doc.createElement('div');
            dot.id = 'revalue-cursor-dot';
            ring = doc.createElement('div');
            ring.id = 'revalue-cursor-ring';

            const style = doc.createElement('style');
            style.id = 'revalue-cursor-style';
            style.textContent = `
                #revalue-cursor-dot {
                    position: fixed;
                    top: 0; left: 0;
                    width: 8px; height: 8px;
                    background: #00e599;
                    border-radius: 50%;
                    pointer-events: none;
                    z-index: 9999999;
                    transform: translate(-50%, -50%);
                    box-shadow: 0 0 14px #00e599, 0 0 22px #d4af37;
                    transition: width 0.15s, height 0.15s, background-color 0.2s;
                }
                #revalue-cursor-ring {
                    position: fixed;
                    top: 0; left: 0;
                    width: 38px; height: 38px;
                    border: 1.5px solid rgba(212, 175, 55, 0.65);
                    border-radius: 50%;
                    pointer-events: none;
                    z-index: 9999998;
                    transform: translate(-50%, -50%);
                    transition: width 0.22s cubic-bezier(0.2, 0.8, 0.2, 1), 
                                height 0.22s cubic-bezier(0.2, 0.8, 0.2, 1), 
                                border-color 0.2s;
                    box-shadow: 0 0 20px rgba(212, 175, 55, 0.25), inset 0 0 10px rgba(0, 229, 153, 0.15);
                }
                .rv-sparkle-trail {
                    position: fixed;
                    width: 4px; height: 4px;
                    border-radius: 50%;
                    pointer-events: none;
                    z-index: 9999997;
                    transform: translate(-50%, -50%);
                    animation: rvTrailSpark 0.55s ease-out forwards;
                }
                @keyframes rvTrailSpark {
                    0% {
                        opacity: 0.9;
                        transform: translate(-50%, -50%) scale(1.4);
                    }
                    100% {
                        opacity: 0;
                        transform: translate(-50%, -50%) scale(0.2);
                    }
                }
            `;
            doc.head.appendChild(style);
            doc.body.appendChild(dot);
            doc.body.appendChild(ring);

            let mouseX = window.innerWidth / 2;
            let mouseY = window.innerHeight / 2;
            let ringX = mouseX;
            let ringY = mouseY;
            let lastSparkleTime = 0;

            doc.addEventListener('mousemove', (e) => {
                mouseX = e.clientX;
                mouseY = e.clientY;
                dot.style.left = mouseX + 'px';
                dot.style.top = mouseY + 'px';

                const now = Date.now();
                if (now - lastSparkleTime > 32) {
                    lastSparkleTime = now;
                    const sparkle = doc.createElement('div');
                    sparkle.className = 'rv-sparkle-trail';
                    sparkle.style.left = mouseX + 'px';
                    sparkle.style.top = mouseY + 'px';
                    const isGold = Math.random() > 0.45;
                    sparkle.style.background = isGold ? '#d4af37' : '#00e599';
                    sparkle.style.boxShadow = isGold ? '0 0 8px #d4af37' : '0 0 8px #00e599';
                    doc.body.appendChild(sparkle);
                    setTimeout(() => { sparkle.remove(); }, 550);
                }
            });

            function renderRing() {
                ringX += (mouseX - ringX) * 0.18;
                ringY += (mouseY - ringY) * 0.18;
                ring.style.left = ringX + 'px';
                ring.style.top = ringY + 'px';
                requestAnimationFrame(renderRing);
            }
            requestAnimationFrame(renderRing);

            // Magnetic highlight expansion when hovering on interactive controls
            doc.addEventListener('mouseover', (e) => {
                const target = e.target.closest('button, select, input, [role="button"], .stSelectbox, .stRadio');
                if (target) {
                    ring.style.width = '56px';
                    ring.style.height = '56px';
                    ring.style.borderColor = '#00e599';
                    ring.style.boxShadow = '0 0 24px rgba(0, 229, 153, 0.4)';
                    dot.style.transform = 'translate(-50%, -50%) scale(1.4)';
                }
            });
            doc.addEventListener('mouseout', (e) => {
                const target = e.target.closest('button, select, input, [role="button"], .stSelectbox, .stRadio');
                if (target) {
                    ring.style.width = '38px';
                    ring.style.height = '38px';
                    ring.style.borderColor = 'rgba(212, 175, 55, 0.65)';
                    ring.style.boxShadow = '0 0 20px rgba(212, 175, 55, 0.25)';
                    dot.style.transform = 'translate(-50%, -50%) scale(1)';
                }
            });
            // ── Real-Time 60fps Battery Slider & Display Card Synchronizer ──
            function syncBatteryDisplay() {
                try {
                    const card = doc.getElementById('revalue-battery-card');
                    const num = doc.getElementById('revalue-battery-number');
                    const badge = doc.getElementById('revalue-battery-badge');
                    if (!card || !num || !badge) return;

                    const sliderThumb = doc.querySelector('div[data-testid="stSlider"] div[role="slider"]');
                    const thumbVal = doc.querySelector('div[data-testid="stThumbValue"]');
                    let val = null;

                    if (sliderThumb && sliderThumb.getAttribute('aria-valuenow')) {
                        val = parseInt(sliderThumb.getAttribute('aria-valuenow'), 10);
                    } else if (thumbVal && thumbVal.textContent) {
                        val = parseInt(thumbVal.textContent.trim(), 10);
                    }

                    if (val !== null && !isNaN(val) && val >= 50 && val <= 100) {
                        const currentText = num.textContent.replace('%', '').trim();
                        if (currentText !== String(val)) {
                            num.textContent = val + '%';
                            const color = val >= 85 ? '#38bdf8' : (val >= 80 ? '#facc15' : '#f87171');
                            const statusText = val >= 85 
                                ? '💎 Optimal Battery Capacity (≥85%) — Zero Degradation'
                                : (val >= 80 
                                    ? '✅ Good Battery Health (≥80%) — Normal Daily Performance' 
                                    : '⚠️ Service Recommended (<80%) — Service Adjustment Applies');
                            
                            card.style.borderColor = color;
                            badge.style.borderColor = color;
                            badge.style.color = color;
                            badge.textContent = statusText;
                            num.style.textShadow = '0 0 25px ' + color;
                        }
                    }
                } catch(e) {}
            }

            doc.addEventListener('pointermove', syncBatteryDisplay, true);
            doc.addEventListener('mousemove', syncBatteryDisplay, true);
            doc.addEventListener('touchmove', syncBatteryDisplay, true);
            doc.addEventListener('input', syncBatteryDisplay, true);
            doc.addEventListener('change', syncBatteryDisplay, true);
            doc.addEventListener('click', syncBatteryDisplay, true);
            setInterval(syncBatteryDisplay, 30);
        }
    } catch(err) {
        // Fallback gracefully
    }
})();
</script>
"""
components.html(CURSOR_JS, height=0, width=0)

# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_resources():
    pipelines, meta = {}, {}
    model_map = {
        "best":              "best_model_pipeline.joblib",
        "ridge":             "ridge_l2_pipeline.joblib",
        "random_forest":     "random_forest_pipeline.joblib",
        "gradient_boosting": "gradient_boosting_pipeline.joblib",
        "unsupervised":      "unsupervised_pipeline.joblib",
    }
    for key, fname in model_map.items():
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            pipelines[key] = joblib.load(p)
    for key, fname in [("bench", "models_benchmark.json"), ("unsup", "unsupervised_summary.json"), ("eval", "evaluation_diagnostics.json")]:
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            with open(p) as f:
                meta[key] = json.load(f)
    train_df = test_df = None
    for split, fname in [("train", "train.csv"), ("test", "test.csv")]:
        p = os.path.join(DATA_DIR, "processed", fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            if split == "train":
                train_df = df
            else:
                test_df = df
    return pipelines, meta, train_df, test_df

pipelines, meta, train_df, test_df = load_all_resources()

DARK_CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#cbd5e1",
    font_family="Plus Jakarta Sans",
    margin=dict(l=15, r=15, t=35, b=15),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.08)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

def style_fig(fig, height=320):
    fig.update_layout(**DARK_CHART_THEME, height=height)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# CURRENCY & INR UTILS (100% RUPEES NATIVE)
# ─────────────────────────────────────────────────────────────────────────────
def usd_to_inr(usd_val):
    return round(float(usd_val) * USD_TO_INR)

def fmt_inr(val):
    if isinstance(val, (float, int)):
        if val >= 100000:
            return f"₹{val/100000:.2f} Lakh"
        return f"₹{int(val):,}"
    return f"₹{val}"

# ─────────────────────────────────────────────────────────────────────────────
# MASSIVE EXPANDED FLAGSHIP CATALOGUE (10 BRANDS, 120+ POPULAR MODELS)
# ─────────────────────────────────────────────────────────────────────────────
POPULAR_PHONES = {
    "Apple": {
        "icon": "🍏",
        "models": {
            "iPhone 16 Pro Max": {"orig_inr": 144900, "screen": 6.9, "ram": 8, "storage": 256, "battery": 99, "launch_year": 2024},
            "iPhone 16 Pro":     {"orig_inr": 119900, "screen": 6.3, "ram": 8, "storage": 128, "battery": 99, "launch_year": 2024},
            "iPhone 16 Plus":    {"orig_inr": 89900,  "screen": 6.7, "ram": 8, "storage": 128, "battery": 98, "launch_year": 2024},
            "iPhone 16":         {"orig_inr": 79900,  "screen": 6.1, "ram": 8, "storage": 128, "battery": 98, "launch_year": 2024},
            "iPhone 15 Pro Max": {"orig_inr": 159900, "screen": 6.7, "ram": 8, "storage": 256, "battery": 94, "launch_year": 2023},
            "iPhone 15 Pro":     {"orig_inr": 134900, "screen": 6.1, "ram": 8, "storage": 128, "battery": 92, "launch_year": 2023},
            "iPhone 15 Plus":    {"orig_inr": 89900,  "screen": 6.7, "ram": 6, "storage": 128, "battery": 93, "launch_year": 2023},
            "iPhone 15":         {"orig_inr": 79900,  "screen": 6.1, "ram": 6, "storage": 128, "battery": 91, "launch_year": 2023},
            "iPhone 14 Pro Max": {"orig_inr": 139900, "screen": 6.7, "ram": 6, "storage": 128, "battery": 88, "launch_year": 2022},
            "iPhone 14 Pro":     {"orig_inr": 119900, "screen": 6.1, "ram": 6, "storage": 128, "battery": 87, "launch_year": 2022},
            "iPhone 14 Plus":    {"orig_inr": 79900,  "screen": 6.7, "ram": 6, "storage": 128, "battery": 88, "launch_year": 2022},
            "iPhone 14":         {"orig_inr": 69900,  "screen": 6.1, "ram": 6, "storage": 128, "battery": 86, "launch_year": 2022},
            "iPhone 13 Pro Max": {"orig_inr": 129900, "screen": 6.7, "ram": 6, "storage": 128, "battery": 84, "launch_year": 2021},
            "iPhone 13 Pro":     {"orig_inr": 109900, "screen": 6.1, "ram": 6, "storage": 128, "battery": 84, "launch_year": 2021},
            "iPhone 13":         {"orig_inr": 59900,  "screen": 6.1, "ram": 4, "storage": 128, "battery": 83, "launch_year": 2021},
            "iPhone 13 Mini":    {"orig_inr": 54900,  "screen": 5.4, "ram": 4, "storage": 128, "battery": 81, "launch_year": 2021},
            "iPhone 12 Pro Max": {"orig_inr": 119900, "screen": 6.7, "ram": 6, "storage": 128, "battery": 80, "launch_year": 2020},
            "iPhone 12 Pro":     {"orig_inr": 99900,  "screen": 6.1, "ram": 6, "storage": 128, "battery": 80, "launch_year": 2020},
            "iPhone 12":         {"orig_inr": 49900,  "screen": 6.1, "ram": 4, "storage": 64,  "battery": 80, "launch_year": 2020},
            "iPhone 11 Pro Max": {"orig_inr": 99900,  "screen": 6.5, "ram": 4, "storage": 64,  "battery": 78, "launch_year": 2019},
            "iPhone 11":         {"orig_inr": 39900,  "screen": 6.1, "ram": 4, "storage": 64,  "battery": 77, "launch_year": 2019},
            "iPhone SE (2022)":  {"orig_inr": 43900,  "screen": 4.7, "ram": 4, "storage": 64,  "battery": 82, "launch_year": 2022},
        }
    },
    "Samsung": {
        "icon": "🌌",
        "models": {
            "Galaxy S24 Ultra":  {"orig_inr": 129999, "screen": 6.8, "ram": 12, "storage": 256, "battery": 97, "launch_year": 2024},
            "Galaxy S24+":       {"orig_inr": 99999,  "screen": 6.7, "ram": 12, "storage": 256, "battery": 96, "launch_year": 2024},
            "Galaxy S24":        {"orig_inr": 79999,  "screen": 6.2, "ram": 8,  "storage": 128, "battery": 95, "launch_year": 2024},
            "Galaxy S23 Ultra":  {"orig_inr": 104999, "screen": 6.8, "ram": 12, "storage": 256, "battery": 90, "launch_year": 2023},
            "Galaxy S23+":       {"orig_inr": 84999,  "screen": 6.6, "ram": 8,  "storage": 256, "battery": 89, "launch_year": 2023},
            "Galaxy S23":        {"orig_inr": 64999,  "screen": 6.1, "ram": 8,  "storage": 128, "battery": 88, "launch_year": 2023},
            "Galaxy S23 FE":     {"orig_inr": 49999,  "screen": 6.4, "ram": 8,  "storage": 128, "battery": 87, "launch_year": 2023},
            "Galaxy S22 Ultra":  {"orig_inr": 84999,  "screen": 6.8, "ram": 12, "storage": 256, "battery": 82, "launch_year": 2022},
            "Galaxy S22+":       {"orig_inr": 69999,  "screen": 6.6, "ram": 8,  "storage": 128, "battery": 82, "launch_year": 2022},
            "Galaxy S22":        {"orig_inr": 57999,  "screen": 6.1, "ram": 8,  "storage": 128, "battery": 81, "launch_year": 2022},
            "Galaxy S21 Ultra":  {"orig_inr": 75999,  "screen": 6.8, "ram": 12, "storage": 256, "battery": 79, "launch_year": 2021},
            "Galaxy S21 FE 5G":  {"orig_inr": 39999,  "screen": 6.4, "ram": 8,  "storage": 128, "battery": 80, "launch_year": 2021},
            "Galaxy Z Fold 6":   {"orig_inr": 164999, "screen": 7.6, "ram": 12, "storage": 256, "battery": 98, "launch_year": 2024},
            "Galaxy Z Flip 6":   {"orig_inr": 109999, "screen": 6.7, "ram": 12, "storage": 256, "battery": 97, "launch_year": 2024},
            "Galaxy Z Fold 5":   {"orig_inr": 154999, "screen": 7.6, "ram": 12, "storage": 256, "battery": 89, "launch_year": 2023},
            "Galaxy Z Flip 5":   {"orig_inr": 99999,  "screen": 6.7, "ram": 8,  "storage": 256, "battery": 88, "launch_year": 2023},
            "Galaxy Z Fold 4":   {"orig_inr": 139999, "screen": 7.6, "ram": 12, "storage": 256, "battery": 84, "launch_year": 2022},
            "Galaxy Z Flip 4":   {"orig_inr": 89999,  "screen": 6.7, "ram": 8,  "storage": 128, "battery": 83, "launch_year": 2022},
            "Galaxy Note 20 Ultra":{"orig_inr": 104999,"screen": 6.9, "ram": 12, "storage": 256, "battery": 78, "launch_year": 2020},
            "Galaxy A55 5G":     {"orig_inr": 39999,  "screen": 6.6, "ram": 8,  "storage": 128, "battery": 94, "launch_year": 2024},
            "Galaxy A54 5G":     {"orig_inr": 38999,  "screen": 6.4, "ram": 8,  "storage": 128, "battery": 85, "launch_year": 2023},
            "Galaxy A35 5G":     {"orig_inr": 30999,  "screen": 6.6, "ram": 8,  "storage": 128, "battery": 92, "launch_year": 2024},
            "Galaxy A34 5G":     {"orig_inr": 25999,  "screen": 6.6, "ram": 8,  "storage": 128, "battery": 88, "launch_year": 2023},
            "Galaxy M34 5G":     {"orig_inr": 18999,  "screen": 6.5, "ram": 6,  "storage": 128, "battery": 86, "launch_year": 2023},
        }
    },
    "OnePlus": {
        "icon": "1️⃣",
        "models": {
            "OnePlus Open Fold": {"orig_inr": 139999, "screen": 7.82, "ram": 16, "storage": 512, "battery": 94, "launch_year": 2023},
            "OnePlus 12":        {"orig_inr": 64999, "screen": 6.82, "ram": 12, "storage": 256, "battery": 95, "launch_year": 2024},
            "OnePlus 12R":       {"orig_inr": 39999, "screen": 6.78, "ram": 8,  "storage": 128, "battery": 94, "launch_year": 2024},
            "OnePlus 11":        {"orig_inr": 56999, "screen": 6.7,  "ram": 8,  "storage": 128, "battery": 89, "launch_year": 2023},
            "OnePlus 11R":       {"orig_inr": 39999, "screen": 6.74, "ram": 8,  "storage": 128, "battery": 87, "launch_year": 2023},
            "OnePlus 10 Pro":    {"orig_inr": 66999, "screen": 6.7,  "ram": 8,  "storage": 128, "battery": 82, "launch_year": 2022},
            "OnePlus 10T":       {"orig_inr": 49999, "screen": 6.7,  "ram": 12, "storage": 128, "battery": 83, "launch_year": 2022},
            "OnePlus 10R":       {"orig_inr": 34999, "screen": 6.7,  "ram": 8,  "storage": 128, "battery": 81, "launch_year": 2022},
            "OnePlus 9 Pro":     {"orig_inr": 54999, "screen": 6.7,  "ram": 8,  "storage": 128, "battery": 79, "launch_year": 2021},
            "OnePlus 9":         {"orig_inr": 44999, "screen": 6.55, "ram": 8,  "storage": 128, "battery": 79, "launch_year": 2021},
            "OnePlus 9RT":       {"orig_inr": 42999, "screen": 6.62, "ram": 8,  "storage": 128, "battery": 80, "launch_year": 2022},
            "OnePlus 8T":        {"orig_inr": 38999, "screen": 6.55, "ram": 8,  "storage": 128, "battery": 76, "launch_year": 2020},
            "OnePlus Nord 4":    {"orig_inr": 29999, "screen": 6.74, "ram": 8,  "storage": 128, "battery": 96, "launch_year": 2024},
            "OnePlus Nord CE 4": {"orig_inr": 24999, "screen": 6.7,  "ram": 8,  "storage": 128, "battery": 95, "launch_year": 2024},
            "OnePlus Nord 3":    {"orig_inr": 33999, "screen": 6.74, "ram": 8,  "storage": 128, "battery": 86, "launch_year": 2023},
            "OnePlus Nord CE 3": {"orig_inr": 26999, "screen": 6.7,  "ram": 8,  "storage": 128, "battery": 88, "launch_year": 2023},
        }
    },
    "Google": {
        "icon": "🔍",
        "models": {
            "Pixel 9 Pro XL":    {"orig_inr": 124999, "screen": 6.8, "ram": 16, "storage": 128, "battery": 99, "launch_year": 2024},
            "Pixel 9 Pro":       {"orig_inr": 109999, "screen": 6.3, "ram": 16, "storage": 128, "battery": 99, "launch_year": 2024},
            "Pixel 9":           {"orig_inr": 79999,  "screen": 6.3, "ram": 12, "storage": 128, "battery": 98, "launch_year": 2024},
            "Pixel 8 Pro":       {"orig_inr": 106999, "screen": 6.7, "ram": 12, "storage": 128, "battery": 93, "launch_year": 2023},
            "Pixel 8":           {"orig_inr": 75999,  "screen": 6.2, "ram": 8,  "storage": 128, "battery": 92, "launch_year": 2023},
            "Pixel 8a":          {"orig_inr": 52999,  "screen": 6.1, "ram": 8,  "storage": 128, "battery": 94, "launch_year": 2024},
            "Pixel 7 Pro":       {"orig_inr": 84999,  "screen": 6.7, "ram": 12, "storage": 128, "battery": 85, "launch_year": 2022},
            "Pixel 7":           {"orig_inr": 59999,  "screen": 6.3, "ram": 8,  "storage": 128, "battery": 84, "launch_year": 2022},
            "Pixel 7a":          {"orig_inr": 43999,  "screen": 6.1, "ram": 8,  "storage": 128, "battery": 86, "launch_year": 2023},
            "Pixel 6 Pro":       {"orig_inr": 69999,  "screen": 6.7, "ram": 12, "storage": 128, "battery": 80, "launch_year": 2021},
            "Pixel 6":           {"orig_inr": 54999,  "screen": 6.4, "ram": 8,  "storage": 128, "battery": 80, "launch_year": 2021},
            "Pixel 6a":          {"orig_inr": 39999,  "screen": 6.1, "ram": 6,  "storage": 128, "battery": 80, "launch_year": 2022},
        }
    },
    "Nothing": {
        "icon": "🎯",
        "models": {
            "Nothing Phone (2)":     {"orig_inr": 44999, "screen": 6.7, "ram": 12, "storage": 256, "battery": 91, "launch_year": 2023},
            "Nothing Phone (2a) Plus":{"orig_inr": 27999, "screen": 6.7, "ram": 8,  "storage": 256, "battery": 96, "launch_year": 2024},
            "Nothing Phone (2a)":    {"orig_inr": 23999, "screen": 6.7, "ram": 8,  "storage": 128, "battery": 95, "launch_year": 2024},
            "Nothing Phone (1)":     {"orig_inr": 32999, "screen": 6.55,"ram": 8,  "storage": 128, "battery": 82, "launch_year": 2022},
            "CMF Phone 1":           {"orig_inr": 15999, "screen": 6.67,"ram": 6,  "storage": 128, "battery": 96, "launch_year": 2024},
        }
    },
    "Xiaomi": {
        "icon": "⚡",
        "models": {
            "Xiaomi 14 Ultra":   {"orig_inr": 99999, "screen": 6.73, "ram": 16, "storage": 512, "battery": 97, "launch_year": 2024},
            "Xiaomi 14":         {"orig_inr": 69999, "screen": 6.36, "ram": 12, "storage": 512, "battery": 95, "launch_year": 2024},
            "Xiaomi 13 Pro":     {"orig_inr": 79999, "screen": 6.73, "ram": 12, "storage": 256, "battery": 88, "launch_year": 2023},
            "Xiaomi 12 Pro":     {"orig_inr": 62999, "screen": 6.73, "ram": 8,  "storage": 256, "battery": 83, "launch_year": 2022},
            "Xiaomi 11T Pro":    {"orig_inr": 39999, "screen": 6.67, "ram": 8,  "storage": 128, "battery": 80, "launch_year": 2021},
            "Redmi Note 13 Pro+":{"orig_inr": 31999, "screen": 6.67, "ram": 8,  "storage": 256, "battery": 92, "launch_year": 2024},
            "Redmi Note 13 Pro": {"orig_inr": 25999, "screen": 6.67, "ram": 8,  "storage": 128, "battery": 91, "launch_year": 2024},
            "Redmi Note 12 Pro+":{"orig_inr": 29999, "screen": 6.67, "ram": 8,  "storage": 256, "battery": 85, "launch_year": 2023},
            "POCO F6 5G":        {"orig_inr": 29999, "screen": 6.67, "ram": 8,  "storage": 256, "battery": 95, "launch_year": 2024},
            "POCO X6 Pro 5G":    {"orig_inr": 24999, "screen": 6.67, "ram": 8,  "storage": 256, "battery": 93, "launch_year": 2024},
            "POCO F5 5G":        {"orig_inr": 27999, "screen": 6.67, "ram": 8,  "storage": 256, "battery": 89, "launch_year": 2023},
            "POCO X5 Pro 5G":    {"orig_inr": 22999, "screen": 6.67, "ram": 8,  "storage": 128, "battery": 84, "launch_year": 2023},
        }
    },
    "Vivo": {
        "icon": "🎵",
        "models": {
            "Vivo X100 Pro":     {"orig_inr": 89999, "screen": 6.78, "ram": 16, "storage": 512, "battery": 96, "launch_year": 2024},
            "Vivo X100":         {"orig_inr": 63999, "screen": 6.78, "ram": 12, "storage": 256, "battery": 94, "launch_year": 2024},
            "Vivo X90 Pro":      {"orig_inr": 84999, "screen": 6.78, "ram": 12, "storage": 256, "battery": 88, "launch_year": 2023},
            "Vivo V30 Pro":      {"orig_inr": 41999, "screen": 6.78, "ram": 8,  "storage": 256, "battery": 93, "launch_year": 2024},
            "Vivo V29 Pro":      {"orig_inr": 39999, "screen": 6.78, "ram": 8,  "storage": 256, "battery": 90, "launch_year": 2023},
            "Vivo V27 Pro":      {"orig_inr": 37999, "screen": 6.78, "ram": 8,  "storage": 128, "battery": 85, "launch_year": 2023},
            "Vivo T3 Pro":       {"orig_inr": 24999, "screen": 6.77, "ram": 8,  "storage": 128, "battery": 95, "launch_year": 2024},
            "iQOO 12 5G":        {"orig_inr": 52999, "screen": 6.78, "ram": 12, "storage": 256, "battery": 94, "launch_year": 2024},
            "iQOO Neo 9 Pro":    {"orig_inr": 35999, "screen": 6.78, "ram": 8,  "storage": 128, "battery": 93, "launch_year": 2024},
            "iQOO Neo 7 Pro":    {"orig_inr": 32999, "screen": 6.78, "ram": 8,  "storage": 128, "battery": 87, "launch_year": 2023},
            "iQOO Z9s Pro":      {"orig_inr": 24999, "screen": 6.77, "ram": 8,  "storage": 128, "battery": 95, "launch_year": 2024},
        }
    },
    "Realme": {
        "icon": "🟡",
        "models": {
            "Realme GT 6":       {"orig_inr": 40999, "screen": 6.78, "ram": 12, "storage": 256, "battery": 97, "launch_year": 2024},
            "Realme GT 6T":      {"orig_inr": 30999, "screen": 6.78, "ram": 8,  "storage": 128, "battery": 96, "launch_year": 2024},
            "Realme 12 Pro+ 5G": {"orig_inr": 29999, "screen": 6.7,  "ram": 8,  "storage": 256, "battery": 92, "launch_year": 2024},
            "Realme 12 Pro 5G":  {"orig_inr": 25999, "screen": 6.7,  "ram": 8,  "storage": 128, "battery": 91, "launch_year": 2024},
            "Realme 11 Pro+ 5G": {"orig_inr": 27999, "screen": 6.7,  "ram": 8,  "storage": 256, "battery": 87, "launch_year": 2023},
            "Realme Narzo 70 Pro":{"orig_inr": 19999, "screen": 6.67, "ram": 8,  "storage": 128, "battery": 92, "launch_year": 2024},
            "Realme GT 2 Pro":   {"orig_inr": 49999, "screen": 6.7,  "ram": 8,  "storage": 128, "battery": 83, "launch_year": 2022},
            "Realme 10 Pro+ 5G": {"orig_inr": 24999, "screen": 6.7,  "ram": 8,  "storage": 128, "battery": 85, "launch_year": 2022},
        }
    },
    "Motorola": {
        "icon": "Ⓜ️",
        "models": {
            "Razr 50 Ultra Fold":{"orig_inr": 89999, "screen": 6.9,  "ram": 12, "storage": 512, "battery": 98, "launch_year": 2024},
            "Razr 40 Ultra":     {"orig_inr": 69999, "screen": 6.9,  "ram": 8,  "storage": 256, "battery": 89, "launch_year": 2023},
            "Edge 50 Ultra":     {"orig_inr": 59999, "screen": 6.7,  "ram": 12, "storage": 512, "battery": 96, "launch_year": 2024},
            "Edge 50 Pro":       {"orig_inr": 31999, "screen": 6.7,  "ram": 8,  "storage": 256, "battery": 93, "launch_year": 2024},
            "Edge 50 Fusion":    {"orig_inr": 22999, "screen": 6.7,  "ram": 8,  "storage": 128, "battery": 92, "launch_year": 2024},
            "Edge 40 Neo":       {"orig_inr": 22999, "screen": 6.55, "ram": 8,  "storage": 128, "battery": 88, "launch_year": 2023},
            "Moto G84 5G":       {"orig_inr": 18999, "screen": 6.55, "ram": 12, "storage": 256, "battery": 87, "launch_year": 2023},
            "Moto G54 5G":       {"orig_inr": 15999, "screen": 6.5,  "ram": 8,  "storage": 128, "battery": 89, "launch_year": 2023},
        }
    },
    "Oppo": {
        "icon": "🌀",
        "models": {
            "Find N3 Flip Fold": {"orig_inr": 94999, "screen": 6.8,  "ram": 12, "storage": 256, "battery": 92, "launch_year": 2023},
            "Reno 12 Pro":       {"orig_inr": 36999, "screen": 6.7,  "ram": 12, "storage": 256, "battery": 95, "launch_year": 2024},
            "Reno 11 Pro":       {"orig_inr": 39999, "screen": 6.7,  "ram": 12, "storage": 256, "battery": 90, "launch_year": 2024},
            "Reno 10 Pro+":      {"orig_inr": 54999, "screen": 6.74, "ram": 12, "storage": 256, "battery": 89, "launch_year": 2023},
            "Reno 8 Pro":        {"orig_inr": 45999, "screen": 6.7,  "ram": 12, "storage": 256, "battery": 83, "launch_year": 2022},
            "F27 Pro+ 5G":       {"orig_inr": 27999, "screen": 6.7,  "ram": 8,  "storage": 256, "battery": 95, "launch_year": 2024},
            "F25 Pro 5G":        {"orig_inr": 23999, "screen": 6.7,  "ram": 8,  "storage": 128, "battery": 92, "launch_year": 2024},
            "F23 5G":            {"orig_inr": 24999, "screen": 6.72, "ram": 8,  "storage": 256, "battery": 87, "launch_year": 2023},
        }
    }
}

POPULAR_LAPTOPS = {
    "Apple": {
        "icon": "🍏",
        "models": {
            "MacBook Air M3":    {"orig_inr": 114900, "screen": 13.6, "ram": 8,  "storage": 256, "battery": 97, "launch_year": 2024},
            "MacBook Air M2":    {"orig_inr": 99900,  "screen": 13.6, "ram": 8,  "storage": 256, "battery": 91, "launch_year": 2022},
            "MacBook Air M1":    {"orig_inr": 79900,  "screen": 13.3, "ram": 8,  "storage": 256, "battery": 84, "launch_year": 2020},
            "MacBook Pro 14 M3": {"orig_inr": 169900, "screen": 14.2, "ram": 16, "storage": 512, "battery": 95, "launch_year": 2023},
            "MacBook Pro 16 M2": {"orig_inr": 229900, "screen": 16.2, "ram": 16, "storage": 512, "battery": 90, "launch_year": 2023},
        }
    },
    "Dell": {
        "icon": "💠",
        "models": {
            "Dell XPS 13 Plus":  {"orig_inr": 149990, "screen": 13.4, "ram": 16, "storage": 512, "battery": 88, "launch_year": 2022},
            "Dell XPS 15":       {"orig_inr": 189990, "screen": 15.6, "ram": 16, "storage": 1000, "battery": 86, "launch_year": 2023},
            "Dell Inspiron 15":  {"orig_inr": 59990,  "screen": 15.6, "ram": 8,  "storage": 512, "battery": 83, "launch_year": 2023},
            "Dell G15 Gaming":   {"orig_inr": 79990,  "screen": 15.6, "ram": 16, "storage": 512, "battery": 85, "launch_year": 2023},
        }
    },
    "HP": {
        "icon": "💙",
        "models": {
            "HP Spectre x360":   {"orig_inr": 139990, "screen": 14.0, "ram": 16, "storage": 512, "battery": 90, "launch_year": 2023},
            "HP Envy 14":        {"orig_inr": 94990,  "screen": 14.0, "ram": 16, "storage": 512, "battery": 86, "launch_year": 2023},
            "HP Pavilion 15":    {"orig_inr": 62990,  "screen": 15.6, "ram": 16, "storage": 512, "battery": 82, "launch_year": 2022},
            "HP Omen 16":        {"orig_inr": 114990, "screen": 16.1, "ram": 16, "storage": 1000, "battery": 87, "launch_year": 2023},
        }
    },
    "Lenovo": {
        "icon": "🔴",
        "models": {
            "ThinkPad X1 Carbon":{"orig_inr": 164990, "screen": 14.0, "ram": 16, "storage": 512, "battery": 89, "launch_year": 2023},
            "Legion Pro 5":      {"orig_inr": 129990, "screen": 16.0, "ram": 16, "storage": 1000, "battery": 88, "launch_year": 2023},
            "IdeaPad Slim 5":    {"orig_inr": 64990,  "screen": 15.6, "ram": 16, "storage": 512, "battery": 85, "launch_year": 2023},
            "Yoga Slim 7":       {"orig_inr": 89990,  "screen": 14.0, "ram": 16, "storage": 512, "battery": 87, "launch_year": 2023},
        }
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "current_tab": "sell_phone",
    "q_step": 1,
    "phone_brand": "Apple",
    "phone_model": "iPhone 16 Pro Max",
    "storage_gb": 256,
    "ram_gb": 8,
    "purchase_year": 2024,
    "orig_price_inr": 144900,
    "power_on": "✅ Yes, powers on normally & battery holds charge",
    "calling_works": "✅ Yes, cellular network & calls work crystal clear",
    "screen_orig": "💎 Original Factory Screen (Never replaced, 100% genuine OEM)",
    "screen_panel_defects": "🟢 Spotless Display (No green lines, white spots, or shadow burn-in)",
    "touch_works": "✅ Flawless Touch (Smooth response across entire display)",
    "screen_cond": "🟢 Flawless: Spotless original glass, zero scratches",
    "b_cond": "✨ Pristine / Like New: No scratches, no dents, looks straight out of the box",
    "buttons_haptics": "✅ All buttons click crisply and haptic vibration works normally",
    "cameras_work": "✅ Both cameras click sharp, focused photos",
    "battery_pct": 99,
    "biometrics_work": "✅ Works instantly",
    "audio_mic": "✅ Loud clear audio on calls & music; mic captures cleanly",
    "charging_port": "✅ Cable fits snugly; fast charging works without angle issues",
    "has_box": True,
    "has_charger": True,
    "has_bill": True,
    "under_warranty": False,
    "l_q_step": 1,
    "laptop_brand": "Apple",
    "laptop_model": "MacBook Air M3",
    "l_ram": 16,
    "l_storage": 512,
    "l_year": 2024,
    "l_orig_inr": 114900,
    "l_screen_cond": "Flawless: Spotless display, zero lines or dead pixels",
    "l_body_cond": "✨ Like New: Spotless lid & chassis, zero dents",
    "l_kb": "✅ All keyboard keys and trackpad click and track smoothly",
    "l_battery": "✅ Battery holds 3+ hours charge",
    "l_ports": "✅ All USB / Type-C ports connect and charge normally",
    "l_charger": True,
    "l_box": True,
    "l_bill": True,
    "estimation_result": None,
    "pickup_booked": False,
    "laptop_estimation_result": None,
    "laptop_pickup_booked": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# ML INFERENCE RUNNER (FIXED ARGUMENT ORDER BUG)
# ─────────────────────────────────────────────────────────────────────────────
def run_model_valuation(input_payload):
    # Ensure all required ML columns exist with correct types and keys
    brand = input_payload.get("brand", "Apple")
    dev_type = input_payload.get("device_type", "Smartphone")
    
    # Auto-detect OS
    if dev_type == "Laptop":
        default_os = "macOS" if brand == "Apple" else "Windows"
    else:
        default_os = "iOS" if brand == "Apple" else "Android"
        
    payload_clean = dict(input_payload)
    if "os" not in payload_clean:
        payload_clean["os"] = default_os
    if "screen_size_inch" not in payload_clean:
        payload_clean["screen_size_inch"] = float(payload_clean.get("screen_size", 6.1))
    if "has_screen_damage" not in payload_clean:
        payload_clean["has_screen_damage"] = int(payload_clean.get("screen_cracked", 0))
    if "has_body_scratches" not in payload_clean:
        payload_clean["has_body_scratches"] = int(payload_clean.get("body_scratched", 0))

    # Standardize condition to valid ordinal levels
    valid_conditions = ["Pristine", "Good", "Fair", "Heavily Used"]
    cond = payload_clean.get("condition", "Good")
    if cond not in valid_conditions:
        if "Pristine" in cond or "Like New" in cond:
            payload_clean["condition"] = "Pristine"
        elif "Good" in cond:
            payload_clean["condition"] = "Good"
        elif "Fair" in cond:
            payload_clean["condition"] = "Fair"
        else:
            payload_clean["condition"] = "Heavily Used"

    df = pd.DataFrame([payload_clean])
    t0 = time.time()
    
    pred_usd = None
    if "best" in pipelines:
        try:
            pred_usd = float(pipelines["best"].predict(df)[0])
        except Exception as pred_err:
            print(f"Prediction fallback notice: {pred_err}")
            
    if pred_usd is None or np.isnan(pred_usd) or pred_usd <= 0:
        orig_usd = float(payload_clean.get("original_price_usd", 800.0))
        age_mos = float(payload_clean.get("age_months", 12.0))
        age_yrs = max(1.0, age_mos / 12.0)
        pred_usd = orig_usd * max(0.20, (1.0 - 0.18 * age_yrs))

    latency_ms = (time.time() - t0) * 1000

    # Ensemble bounds
    preds = []
    for k in ["ridge", "random_forest", "gradient_boosting"]:
        if k in pipelines:
            try:
                preds.append(float(pipelines[k].predict(df)[0]))
            except:
                pass
    if preds:
        lo_usd = max(0.0, float(np.percentile(preds, 15)))
        hi_usd = float(np.percentile(preds, 85))
    else:
        lo_usd = pred_usd * 0.88
        hi_usd = pred_usd * 1.12

    orig_usd = payload_clean.get("original_price_usd", pred_usd / 0.6)
    deprec_pct = round(max(0, (1 - pred_usd / max(orig_usd, 1))) * 100, 1)

    is_anomaly = False
    if "unsupervised" in pipelines:
        try:
            is_anomaly = bool(pipelines["unsupervised"].predict(df)[0] == -1)
        except:
            pass

    # FIXED: Use explicit keyword arguments to avoid positional parameter mismatch
    try:
        log_prediction(
            input_data=payload_clean,
            predicted_price=pred_usd,
            lower_bound=lo_usd,
            upper_bound=hi_usd,
            is_anomaly=1 if is_anomaly else 0,
            latency_ms=latency_ms,
            active_model="Gradient_Boosting"
        )
    except Exception as log_err:
        print(f"Monitoring log notice: {log_err}")

    return {
        "pred_inr": usd_to_inr(pred_usd),
        "lo_inr": usd_to_inr(lo_usd),
        "hi_inr": usd_to_inr(hi_usd),
        "deprec_pct": deprec_pct,
        "is_anomaly": is_anomaly,
        "latency_ms": latency_ms,
        "payload": payload_clean
    }

# ─────────────────────────────────────────────────────────────────────────────
# TOP NAVBAR (REVALUE ELITE BRANDING)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="revalue-nav">
    <div class="revalue-logo">
        <span>👑 ReValue<span style="color:#d4af37;">AI</span></span>
        <span class="revalue-logo-badge">Black Edition</span>
    </div>
    <div class="revalue-trust-tag">
        <span>✨ Official Guaranteed Trade-In &bull; Free Concierge Pickup &bull; Instant UPI Payout</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs styled as luxury pills
nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns(5)
with nav_c1:
    if st.button("📱 Sell Phone", key="nav_phone", use_container_width=True, 
                 type="primary" if st.session_state.current_tab == "sell_phone" else "secondary"):
        st.session_state.current_tab = "sell_phone"
        st.rerun()
with nav_c2:
    if st.button("💻 Sell Laptop", key="nav_laptop", use_container_width=True,
                 type="primary" if st.session_state.current_tab == "sell_laptop" else "secondary"):
        st.session_state.current_tab = "sell_laptop"
        st.rerun()
with nav_c3:
    if st.button("📊 ML Insights", key="nav_insights", use_container_width=True,
                 type="primary" if st.session_state.current_tab == "insights" else "secondary"):
        st.session_state.current_tab = "insights"
        st.rerun()
with nav_c4:
    if st.button("⚙️ MLOps Centre", key="nav_mlops", use_container_width=True,
                 type="primary" if st.session_state.current_tab == "mlops" else "secondary"):
        st.session_state.current_tab = "mlops"
        st.rerun()
with nav_c5:
    if st.button("🎓 Syllabus Coverage", key="nav_syllabus", use_container_width=True,
                 type="primary" if st.session_state.current_tab == "syllabus" else "secondary"):
        st.session_state.current_tab = "syllabus"
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: SELL PHONE (COMPREHENSIVE LUXURY APPRAISAL FLOW)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.current_tab == "sell_phone":
    render_phone_wizard(POPULAR_PHONES, run_model_valuation, fmt_inr)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: SELL LAPTOP (MACBOOK & WINDOWS LAPTOPS)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_tab == "sell_laptop":
    render_laptop_wizard(POPULAR_LAPTOPS, run_model_valuation, fmt_inr)


# TAB 3: ML MODEL INSIGHTS & BENCHMARKS (ALL METRICS IN INR)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_tab == "insights":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-badge"><span class="hero-badge-dot"></span>ML Architecture & Evaluation</div>
        <div class="hero-title">Model Benchmarks & <span>Price Drivers</span></div>
        <div class="hero-sub">Empirical evaluation across 6 regression algorithms on held-out test data. All error metrics converted to Indian Rupees (₹).</div>
    </div>
    """, unsafe_allow_html=True)

    bench = meta.get("bench", {})
    comp = bench.get("models_comparison", {})

    t1, t2, t3, t4 = st.tabs(["Algorithm Comparison (₹)", "Actual vs Predicted (₹)", "Feature Importances", "Residual Analysis (₹)"])

    with t1:
        if comp:
            rows = []
            for name, d in comp.items():
                tm = d.get("test_metrics", {})
                is_champ = (name == bench.get("best_model", "Gradient_Boosting"))
                rows.append({
                    "Algorithm": ("🏆 " if is_champ else "") + name.replace("_", " "),
                    "Test MAE (₹)": f"₹{usd_to_inr(tm.get('MAE', 0)):,}",
                    "Test RMSE (₹)": f"₹{usd_to_inr(tm.get('RMSE', 0)):,}",
                    "R² Score": f"{tm.get('R2_Score', 0):.4f}",
                    "MAPE": f"{tm.get('MAPE', 0):.2f}%",
                    "Deployment Role": "Champion Production" if is_champ else "Challenger"
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            m_names = [k.replace("_", " ") for k in comp.keys()]
            m_mae_inr = [usd_to_inr(comp[k].get("test_metrics", {}).get("MAE", 0)) for k in comp.keys()]
            m_r2 = [float(comp[k].get("test_metrics", {}).get("R2_Score", 0)) for k in comp.keys()]

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(x=m_names, y=m_mae_inr, color=m_mae_inr, 
                             color_continuous_scale=["#0a1628", "#d4af37"],
                             labels={"x": "Model", "y": "Mean Absolute Error (₹)"},
                             title="Test MAE by Model (Lower is Better, ₹)")
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(x=m_names, y=m_r2, color=m_r2,
                             color_continuous_scale=["#0a1628", "#00e599"],
                             labels={"x": "Model", "y": "R² Explained Variance"},
                             title="R² Score by Model (Higher is Better)")
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)

    with t2:
        if test_df is not None and "best" in pipelines:
            try:
                feat_cols = [c for c in test_df.columns if c != "resale_price_usd"]
                y_true_inr = test_df["resale_price_usd"].values * USD_TO_INR
                y_pred_inr = pipelines["best"].predict(test_df[feat_cols]) * USD_TO_INR
                res_inr = y_pred_inr - y_true_inr

                c1, c2 = st.columns(2)
                with c1:
                    fig = px.scatter(x=y_true_inr, y=y_pred_inr, opacity=0.45,
                                     color_discrete_sequence=["#00e599"],
                                     labels={"x": "Actual Resale Price (₹)", "y": "Predicted Price (₹)"},
                                     title="Actual vs Predicted Resale Value (₹)")
                    fig.add_shape(type="line", line=dict(dash="dash", color="#ef4444", width=2),
                                  x0=y_true_inr.min(), y0=y_true_inr.min(), x1=y_true_inr.max(), y1=y_true_inr.max())
                    style_fig(fig)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.histogram(x=res_inr, nbins=40, color_discrete_sequence=["#d4af37"],
                                       labels={"x": "Valuation Error (₹)"},
                                       title="Residual Error Distribution (₹)")
                    style_fig(fig)
                    st.plotly_chart(fig, use_container_width=True)

                mae_val = float(np.mean(np.abs(res_inr)))
                rmse_val = float(np.sqrt(np.mean(res_inr**2)))
                r2_val = float(1 - np.var(res_inr) / np.var(y_true_inr))
                st.markdown(f"""
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 16px;">
                    <div class="kpi-tile"><div class="kpi-lbl">Held-out Test MAE</div><div class="kpi-val">₹{mae_val:,.0f}</div></div>
                    <div class="kpi-tile"><div class="kpi-lbl">Held-out Test RMSE</div><div class="kpi-val">₹{rmse_val:,.0f}</div></div>
                    <div class="kpi-tile"><div class="kpi-lbl">Test R² Score</div><div class="kpi-val">{r2_val:.4f}</div></div>
                    <div class="kpi-tile"><div class="kpi-lbl">Target Currency</div><div class="kpi-val">INR (₹)</div></div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Error computing diagnostics: {e}")

    with t3:
        p = pipelines.get("gradient_boosting") or pipelines.get("best")
        if p and train_df is not None:
            try:
                reg = p.named_steps.get("regressor") or p[-1]
                imp = reg.feature_importances_
                fc = [c for c in train_df.columns if c != "resale_price_usd"]
                n = min(12, len(fc), len(imp))
                idx = np.argsort(imp)[-n:]
                fn = [fc[i] if i < len(fc) else f"feature_{i}" for i in idx]
                fi = [float(imp[i]) for i in idx]
                fig = px.bar(x=fi, y=fn, orientation="h", color=fi,
                             color_continuous_scale=["#0a1628", "#00e599"],
                             title="Top Feature Importances in Resale Valuation")
                style_fig(fig, height=380)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"Feature importance unavailable: {e}")

    with t4:
        if test_df is not None and "best" in pipelines:
            try:
                fc = [c for c in test_df.columns if c != "resale_price_usd"]
                yt = test_df["resale_price_usd"].values * USD_TO_INR
                yp = pipelines["best"].predict(test_df[fc]) * USD_TO_INR
                res = yp - yt
                fig = px.scatter(x=yp, y=res, opacity=0.4, color_discrete_sequence=["#06b6d4"],
                                 labels={"x": "Predicted Price (₹)", "y": "Residual (₹)"},
                                 title="Homoscedasticity Check: Residuals vs Fitted (₹)")
                fig.add_hline(y=0, line=dict(dash="dash", color="#ef4444", width=1.5))
                style_fig(fig)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Residual analysis: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: MLOPS COMMAND CENTRE & DRIFT MONITORING
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_tab == "mlops":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-badge"><span class="hero-badge-dot"></span>Production Observability (M6)</div>
        <div class="hero-title">MLOps & <span>Live Drift Telemetry</span></div>
        <div class="hero-sub">Kolmogorov-Smirnov statistical feature drift monitoring, inference SLA latency tracking, and automatic champion retraining.</div>
    </div>
    """, unsafe_allow_html=True)

    health = get_model_health_summary()
    lat = compute_latency_stats()
    n_logs = get_log_count()
    logs_df = get_all_logs_df()
    drift = {}
    if train_df is not None and logs_df is not None and len(logs_df) >= 5:
        try:
            drift_res = detect_data_drift(train_df, logs_df, min_samples=5)
            drift = drift_res.get("feature_details", {})
        except Exception:
            drift = {}
    
    if not drift and os.path.exists(os.path.join(MODELS_DIR, "latest_drift_report.json")):
        try:
            with open(os.path.join(MODELS_DIR, "latest_drift_report.json"), "r") as f:
                d_snap = json.load(f)
                drift = d_snap.get("feature_details", {})
        except Exception:
            pass

    sev = health.get("severity", "STABLE")

    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px;">
        <div class="kpi-tile">
            <div class="kpi-lbl">Health Score</div>
            <div class="kpi-val">{health.get('health_score', 100):.0f}/100</div>
            <div class="kpi-sub">Status: <span style="color:#00e599; font-weight:700;">{sev}</span></div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-lbl">Total Logged Inferences</div>
            <div class="kpi-val" style="color:#ffffff;">{n_logs:,}</div>
            <div class="kpi-sub">All in INR (₹)</div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-lbl">P50 Latency</div>
            <div class="kpi-val" style="color:#ffffff;">{lat.get('p50_ms', 0):.1f}ms</div>
            <div class="kpi-sub">Median response</div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-lbl">P95 SLA Target</div>
            <div class="kpi-val">{lat.get('p95_ms', 0):.1f}ms</div>
            <div class="kpi-sub">SLA Threshold: &lt;{SLA_LATENCY_P95_MS}ms</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.tabs(["SLA & Live Logs", "Data Drift Report", "Champion Retraining"])

    with m1:
        if n_logs > 0:
            logs = get_recent_logs(25)
            if logs:
                df_logs = pd.DataFrame(logs)
                if "predicted_price_usd" in df_logs.columns:
                    df_logs["Price (₹)"] = (df_logs["predicted_price_usd"] * USD_TO_INR).round(0).astype(int).apply(lambda x: f"₹{x:,}")
                show_cols = [c for c in ["timestamp", "device_type", "brand", "model_name", "condition", "Price (₹)", "is_anomaly", "latency_ms"] if c in df_logs.columns]
                st.dataframe(df_logs[show_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No inference logs recorded yet. Perform valuations in the Sell Phone tab to populate live MLOps data.")

    with m2:
        if drift:
            d_rows = []
            for feat, d in drift.items():
                if isinstance(d, dict):
                    ks_stat = d.get('ks_statistic')
                    p_val = d.get('p_value')
                    psi_val = d.get('psi')
                    d_rows.append({
                        "Feature": feat,
                        "KS Statistic": f"{ks_stat:.4f}" if ks_stat is not None else "N/A",
                        "P-Value": f"{p_val:.4f}" if p_val is not None else "N/A",
                        "PSI": f"{psi_val:.4f}" if psi_val is not None else "N/A",
                        "Drift Status": d.get("severity", d.get("status", "STABLE"))
                    })
            if d_rows:
                st.dataframe(pd.DataFrame(d_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Drift detection initialized. Continuous KS & PSI monitoring active on baseline.")
        else:
            st.info("Drift detection ready. Continuous KS & PSI monitoring active on production baseline.")

    with m3:
        st.markdown("#### Champion / Challenger Retraining Pipeline")
        st.caption("Trigger automated retraining on latest feedback samples. Compares against current Gradient Boosting champion.")
        if st.button("Trigger Pipeline Retraining Now", key="btn_trigger_retrain"):
            with st.spinner("Executing pipeline retrain with cross-validation..."):
                try:
                    res_msg = trigger_retraining()
                    st.success(f"Retraining Completed: {res_msg}")
                    st.cache_resource.clear()
                except Exception as e:
                    st.error(f"Retraining encountered an error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: COURSE SYLLABUS COVERAGE (COURSE OUTCOMES CO-1 TO CO-6)
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.current_tab == "syllabus":
    render_syllabus_page()

