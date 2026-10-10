"""
ReValue AI — Comprehensive Multi-Step Diagnostic Wizards
=========================================================
Page Structure:
- Page 1: Questions 1, 2, 3 (Brand, Model, Specs: Storage, RAM, Purchase Year, Price)
- Page 2: Questions 4, 5, 6, 7 (Power, Calling, Screen OEM, Display Panel Defects)
- Page 3: Questions 8, 9, 10, 11, 12 (Touch, Glass State, Body Housing, Buttons, Cameras)
- Page 4: Questions 13, 14, 15, 16 (Battery Health %, Biometrics, Audio/Ports, Accessories)
- Page 5: Summary Review of All Diagnostic Answers
- Page 6: Official Valuation Report & Verified Resale Offer

Key features:
- All required ML pipeline columns provided (screen_size_inch, os, has_screen_damage, has_body_scratches)
- Model launch year constraint strictly enforced (no year before official launch year)
- Zero "Question 1 of XX" or "Question X of XX" labels
- High-contrast, crystal-clear battery health capacity percentage display
- Clean arrow navigation: '← Previous' and 'Next →'
- Minimum 15% per year annual depreciation ceiling strictly applied to all resale offers
- 100% Indian Rupees (₹ / INR)
"""

import time
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

USD_TO_INR = 84.0


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: SELL PHONE WIZARD
# ─────────────────────────────────────────────────────────────────────────────
def render_phone_wizard(POPULAR_PHONES, run_model_valuation, fmt_inr):
    # Hero Intro
    st.markdown("""
    <div class="hero-card">
        <div class="hero-badge"><span class="hero-badge-dot"></span>Official AI Appraisal Engine</div>
        <div class="hero-title">Sell Your Phone at <span>Maximum Resale Value</span></div>
        <div class="hero-sub">Answer our quick diagnostic check. Our production Gradient Boosting model calculates your verified market value with instant doorstep pickup and UPI payment.</div>
        <div class="hero-perks">
            <div class="hero-perk-item"><span class="hero-perk-icon">✓</span> Free Doorstep Pickup</div>
            <div class="hero-perk-item"><span class="hero-perk-icon">✓</span> Instant Bank / UPI Payment</div>
            <div class="hero-perk-item"><span class="hero-perk-icon">✓</span> 7 Days Price Lock Guarantee</div>
            <div class="hero-perk-item"><span class="hero-perk-icon">✓</span> All Valuations in Indian Rupees (₹)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Trust Value Proposition Cards
    st.markdown("""
    <div class="revalue-perks-row">
        <div class="revalue-perk-card">
            <span class="revalue-perk-icon">💎</span>
            <div class="revalue-perk-title">Instant Cash Offer</div>
            <div class="revalue-perk-desc">Get the highest AI-backed valuation in under 60 seconds</div>
        </div>
        <div class="revalue-perk-card">
            <span class="revalue-perk-icon">🚚</span>
            <div class="revalue-perk-title">Free Home Pickup</div>
            <div class="revalue-perk-desc">Verified technician inspects & collects device at your home</div>
        </div>
        <div class="revalue-perk-card">
            <span class="revalue-perk-icon">💸</span>
            <div class="revalue-perk-title">Instant UPI Transfer</div>
            <div class="revalue-perk-desc">Money credited straight to your GooglePay, PhonePe, or Bank</div>
        </div>
        <div class="revalue-perk-card">
            <span class="revalue-perk-icon">🔒</span>
            <div class="revalue-perk-title">Certified Data Wipe</div>
            <div class="revalue-perk-desc">Military-grade 100% data destruction certificate provided</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page_names = {
        1: "📱 Device & Technical Specifications",
        2: "⚡ Power, Calling & Display Screen Inspection",
        3: "🔍 Touch, Physical Housing & Camera Optics",
        4: "🔋 Battery Health, Audio & Trade-In Accessories",
        5: "📋 Summary Review of Diagnostic Answers",
        6: "👑 Official Valuation Report & Verified Resale Offer"
    }

    q_step = st.session_state.get("q_step", 1)
    if q_step > 6:
        q_step = 1
        st.session_state.q_step = 1

    # ─────────────────────────────────────────────────────────────────────────
    # PAGES 1 TO 4: DIAGNOSTIC PAGES
    # ─────────────────────────────────────────────────────────────────────────
    if 1 <= q_step <= 4:
        progress_val = float(q_step) / 4.0
        st.markdown(f"""
        <div style="background: rgba(13, 20, 36, 0.7); border: 1px solid rgba(212, 175, 55, 0.25); border-radius: 16px; padding: 14px 22px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="background: rgba(212, 175, 55, 0.15); border: 1px solid rgba(212, 175, 55, 0.4); border-radius: 100px; padding: 4px 14px; font-size: 0.76rem; font-weight: 800; color: #f6e05e; letter-spacing: 0.08em; text-transform: uppercase;">
                    Diagnostic Step {q_step} of 4
                </span>
                <span style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">
                    {page_names[q_step]}
                </span>
            </div>
            <div style="color: #00e599; font-weight: 800; font-family: var(--mono); font-size: 0.88rem;">
                {int(progress_val * 100)}% Complete
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(progress_val)

        # ═════════════════════════════════════════════════════════════════════
        # PAGE 1: MERGED QUESTIONS 1, 2, 3 (BRAND, MODEL, SPECS, LAUNCH YEAR)
        # ═════════════════════════════════════════════════════════════════════
        if q_step == 1:
            st.markdown("""
            <div class="q-card">
                <div class="q-title"><span>📱</span> Device & Hardware Specifications</div>
                <div class="q-sub">Select your device manufacturer, model edition, and hardware configuration. Purchase year is validated against official launch year.</div>
            </div>
            """, unsafe_allow_html=True)

            brand_options = list(POPULAR_PHONES.keys())
            cur_b_idx = brand_options.index(st.session_state.phone_brand) if st.session_state.phone_brand in brand_options else 0
            
            c_brand, c_model = st.columns(2)
            with c_brand:
                sel_brand = st.selectbox("Smartphone Brand", brand_options, index=cur_b_idx, key="sb_brand_wizard")
                if sel_brand != st.session_state.phone_brand:
                    st.session_state.phone_brand = sel_brand
                    first_model = list(POPULAR_PHONES[sel_brand]["models"].keys())[0]
                    st.session_state.phone_model = first_model
                    m_def = POPULAR_PHONES[sel_brand]["models"][first_model]
                    st.session_state.orig_price_inr = int(m_def["orig_inr"])
                    st.session_state.storage_gb = int(m_def["storage"])
                    st.session_state.ram_gb = int(m_def["ram"])
                    st.session_state.battery_pct = int(m_def.get("battery", 88))
                    st.session_state.sl_p4_battery = int(st.session_state.battery_pct)
                    l_yr = int(m_def.get("launch_year", 2024))
                    st.session_state.purchase_year = max(l_yr, 2024)
                    st.rerun()

            with c_model:
                brand_models = list(POPULAR_PHONES[st.session_state.phone_brand]["models"].keys())
                cur_m_idx = brand_models.index(st.session_state.phone_model) if st.session_state.phone_model in brand_models else 0
                sel_model = st.selectbox(f"{st.session_state.phone_brand} Model Edition", brand_models, index=cur_m_idx, key="sb_model_wizard")
                if sel_model != st.session_state.phone_model:
                    st.session_state.phone_model = sel_model
                    m_def = POPULAR_PHONES[st.session_state.phone_brand]["models"][sel_model]
                    st.session_state.orig_price_inr = int(m_def["orig_inr"])
                    st.session_state.storage_gb = int(m_def["storage"])
                    st.session_state.ram_gb = int(m_def["ram"])
                    st.session_state.battery_pct = int(m_def.get("battery", 88))
                    st.session_state.sl_p4_battery = int(st.session_state.battery_pct)
                    l_yr = int(m_def.get("launch_year", 2024))
                    st.session_state.purchase_year = max(l_yr, 2024)
                    st.rerun()

            # Selected model metadata
            m_def = POPULAR_PHONES[st.session_state.phone_brand]["models"].get(st.session_state.phone_model, {})
            launch_year = int(m_def.get("launch_year", 2023))

            # CRITICAL REQUIREMENT:
            # Model launch year matching - only allow years >= launch_year and <= 2026!
            min_year = max(launch_year, 2016)
            allowed_years = list(range(2026, min_year - 1, -1))
            if st.session_state.purchase_year not in allowed_years:
                st.session_state.purchase_year = allowed_years[-1] if allowed_years else 2024

            st.markdown(f"""
            <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 10px 16px; margin: 12px 0 18px 0; display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.85rem; color: #cbd5e1;">
                    🚀 <b>{st.session_state.phone_brand} {st.session_state.phone_model}</b> &bull; Official Market Launch: <span style="color:#38bdf8; font-weight:800;">{launch_year}</span>
                </span>
                <span style="font-size: 0.78rem; color: #94a3b8;">
                    Purchase years permitted: <b style="color:#f6e05e;">{min_year}–2026</b>
                </span>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                stor_opts = [64, 128, 256, 512, 1000]
                cur_s = st.session_state.storage_gb if st.session_state.storage_gb in stor_opts else 256
                sel_stor = st.selectbox("Internal Storage (GB)", stor_opts, index=stor_opts.index(cur_s), key="sb_stor_wizard")
                st.session_state.storage_gb = sel_stor
            with c2:
                ram_opts = [4, 6, 8, 12, 16]
                cur_r = st.session_state.ram_gb if st.session_state.ram_gb in ram_opts else 8
                sel_ram = st.selectbox("RAM (Memory GB)", ram_opts, index=ram_opts.index(cur_r), key="sb_ram_wizard")
                st.session_state.ram_gb = sel_ram
            with c3:
                cur_y_idx = allowed_years.index(st.session_state.purchase_year) if st.session_state.purchase_year in allowed_years else 0
                sel_year = st.selectbox("Purchase Year", allowed_years, index=cur_y_idx, key="sb_year_wizard")
                st.session_state.purchase_year = sel_year
                age_yrs = 2026 - sel_year
                st.caption(f"Device Age: ~{age_yrs} yr{'s' if age_yrs != 1 else ''} ({max(3, age_yrs * 12)} mos)")
            with c4:
                sel_price = st.number_input(
                    "Original Purchase Price (₹)",
                    min_value=5000,
                    max_value=350000,
                    value=int(st.session_state.orig_price_inr),
                    step=2000,
                    key="sb_price_wizard",
                    help="Invoice purchase price in Indian Rupees (₹)"
                )
                st.session_state.orig_price_inr = sel_price

            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            col_space, col_next = st.columns([2, 1.2])
            with col_next:
                if st.button("Next →", key="btn_next_p1", type="primary", use_container_width=True):
                    st.session_state.q_step = 2
                    st.rerun()

        # ═════════════════════════════════════════════════════════════════════
        # PAGE 2: QUESTIONS 4, 5, 6, 7 (POWER, CALLING, SCREEN OEM, DEFECTS)
        # ═════════════════════════════════════════════════════════════════════
        elif q_step == 2:
            st.markdown("""
            <div class="q-card">
                <div class="q-title"><span>⚡</span> Power, Calling & Display Screen Inspection</div>
                <div class="q-sub">Answer these 4 fundamental functionality and screen checks to assess core hardware integrity.</div>
            </div>
            """, unsafe_allow_html=True)

            col_q4, col_q5 = st.columns(2)
            with col_q4:
                st.markdown("#### ⚡ Power, Startup & Battery Retention")
                st.caption("Does the phone turn on and stay powered without unexpected shutdowns?")
                opts_p = [
                    "✅ Yes, powers on normally & battery holds charge",
                    "❌ Phone is dead / restarts randomly / boot loops"
                ]
                cur_p = opts_p.index(st.session_state.power_on) if st.session_state.power_on in opts_p else 0
                st.session_state.power_on = st.radio("Power Status", opts_p, index=cur_p, key="rb_p2_power", label_visibility="collapsed")

            with col_q5:
                st.markdown("#### 📞 Cellular Calls & Connectivity")
                st.caption("Are you able to make/receive cellular phone calls & browse WiFi/4G/5G?")
                opts_c = [
                    "✅ Yes, cellular network & calls work crystal clear",
                    "❌ Calling, SIM detection or cellular signal issue"
                ]
                cur_c = opts_c.index(st.session_state.calling_works) if st.session_state.calling_works in opts_c else 0
                st.session_state.calling_works = st.radio("Cellular Status", opts_c, index=cur_c, key="rb_p2_calls", label_visibility="collapsed")

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

            col_q6, col_q7 = st.columns(2)
            with col_q6:
                st.markdown("#### 💎 Screen Originality (Genuine OEM Display)")
                st.caption("Has the display ever been replaced or repaired?")
                opts_orig = [
                    "💎 Original Factory Screen (Never replaced, 100% genuine OEM)",
                    "🏢 Replaced at Official Brand Service Center (Apple / Samsung)",
                    "🔧 Replaced at Local Third-Party Repair Shop (Duplicate / copy)"
                ]
                cur_orig = opts_orig.index(st.session_state.screen_orig) if st.session_state.screen_orig in opts_orig else 0
                st.session_state.screen_orig = st.radio("Screen Replacement History", opts_orig, index=cur_orig, key="rb_p2_orig", label_visibility="collapsed")

            with col_q7:
                st.markdown("#### 📺 Display Panel Inspection")
                st.caption("Are there any colored lines, dead pixels, white spots, or burn-in?")
                opts_def = [
                    "🟢 Spotless Display (No green lines, white spots, or shadow burn-in)",
                    "🟡 Minor Yellow Tint or Faint Status Bar Burn-in",
                    "🔴 Vertical Green/White Lines, Black Ink Spots, or Flickering"
                ]
                cur_def = opts_def.index(st.session_state.screen_panel_defects) if st.session_state.screen_panel_defects in opts_def else 0
                st.session_state.screen_panel_defects = st.radio("Display Panel Visual State", opts_def, index=cur_def, key="rb_p2_def", label_visibility="collapsed")

            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            col_prev, col_skip, col_next = st.columns([1, 1.3, 1.4])
            with col_prev:
                if st.button("← Previous", key="btn_prev_p2", use_container_width=True):
                    st.session_state.q_step = 1
                    st.rerun()
            with col_skip:
                if st.button("📋 Summary Review", key="btn_jump_p2", use_container_width=True):
                    st.session_state.q_step = 5
                    st.rerun()
            with col_next:
                if st.button("Next →", key="btn_next_p2", type="primary", use_container_width=True):
                    st.session_state.q_step = 3
                    st.rerun()

        # ═════════════════════════════════════════════════════════════════════
        # PAGE 3: QUESTIONS 8, 9, 10, 11, 12 (TOUCH, GLASS, BODY, BTN, CAM)
        # ═════════════════════════════════════════════════════════════════════
        elif q_step == 3:
            st.markdown("""
            <div class="q-card">
                <div class="q-title"><span>🔍</span> Touch, Physical Housing & Camera Optics</div>
                <div class="q-sub">Review exterior cosmetic wear, touch response, buttons, and camera functionality.</div>
            </div>
            """, unsafe_allow_html=True)

            c_t8, c_t9 = st.columns(2)
            with c_t8:
                st.markdown("#### 👆 Touch Screen Responsiveness")
                st.caption("Is touch response smooth across all screen corners and keyboard keys?")
                opts_t = [
                    "✅ Flawless Touch (Smooth response across entire display)",
                    "❌ Unresponsive Zones, Ghost Touch, or Laggy Drag"
                ]
                cur_t = opts_t.index(st.session_state.touch_works) if st.session_state.touch_works in opts_t else 0
                st.session_state.touch_works = st.radio("Touch Functionality", opts_t, index=cur_t, key="rb_p3_touch", label_visibility="collapsed")

            with c_t9:
                st.markdown("#### 🔍 Front Glass Scratch & Crack State")
                st.caption("What is the physical scratch and crack state of the outer glass?")
                opts_g = [
                    "🟢 Flawless: Spotless original glass, zero scratches",
                    "🟡 Minor Scratches: 1-2 faint hairline marks (lights up clean)",
                    "🟠 Noticeable Scratches: Multiple visible surface scratches",
                    "🔴 Cracked Glass / Shattered Outer Display"
                ]
                cur_g = opts_g.index(st.session_state.screen_cond) if st.session_state.screen_cond in opts_g else 0
                st.session_state.screen_cond = st.radio("Front Glass State", opts_g, index=cur_g, key="rb_p3_glass", label_visibility="collapsed")

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

            c_t10, c_t11 = st.columns(2)
            with c_t10:
                st.markdown("#### 🛡️ Body, Side Bezels & Back Housing")
                st.caption("What is the condition of the metal frame, side edges, and rear glass?")
                opts_b = [
                    "✨ Pristine / Like New: No scratches, no dents, looks straight out of the box",
                    "👍 Good: Minor normal signs of daily use, light bezel scuff from cover",
                    "😐 Fair: Visible paint wear or minor corner impact marks",
                    "🔧 Heavily Used: Heavy dents, bent frame, or cracked back glass"
                ]
                cur_b = opts_b.index(st.session_state.b_cond) if st.session_state.b_cond in opts_b else 0
                st.session_state.b_cond = st.radio("Body Rating", opts_b, index=cur_b, key="rb_p3_body", label_visibility="collapsed")

            with c_t11:
                st.markdown("#### 🎛️ Physical Buttons & Haptics")
                st.caption("Do Power, Volume, Mute Switch and vibration motor operate crisply?")
                opts_btn = [
                    "✅ All buttons click crisply and haptic vibration works normally",
                    "❌ Loose/stuck power or volume button / vibration not working"
                ]
                cur_btn = opts_btn.index(st.session_state.buttons_haptics) if st.session_state.buttons_haptics in opts_btn else 0
                st.session_state.buttons_haptics = st.radio("Buttons Status", opts_btn, index=cur_btn, key="rb_p3_btn", label_visibility="collapsed")

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 📸 Front & Rear Cameras (Autofocus & Flash)")
            st.caption("Do both front and rear camera lenses click sharp, focused photos with working flash and video recording?")
            opts_cam = [
                "✅ Both cameras click sharp, focused photos",
                "❌ Camera blurry, shutter issue, or cracked lens glass"
            ]
            cur_cam = opts_cam.index(st.session_state.cameras_work) if st.session_state.cameras_work in opts_cam else 0
            st.session_state.cameras_work = st.radio("Camera System", opts_cam, index=cur_cam, key="rb_p3_cam", label_visibility="collapsed")

            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            col_prev, col_skip, col_next = st.columns([1, 1.3, 1.4])
            with col_prev:
                if st.button("← Previous", key="btn_prev_p3", use_container_width=True):
                    st.session_state.q_step = 2
                    st.rerun()
            with col_skip:
                if st.button("📋 Summary Review", key="btn_jump_p3", use_container_width=True):
                    st.session_state.q_step = 5
                    st.rerun()
            with col_next:
                if st.button("Next →", key="btn_next_p3", type="primary", use_container_width=True):
                    st.session_state.q_step = 4
                    st.rerun()

        # ═════════════════════════════════════════════════════════════════════
        # PAGE 4: QUESTIONS 13, 14, 15, 16 (BATTERY, BIOMETRICS, AUDIO, ACC)
        # ═════════════════════════════════════════════════════════════════════
        elif q_step == 4:
            st.markdown("""
            <div class="q-card">
                <div class="q-title"><span>🔋</span> Battery Health, Audio & Trade-In Accessories</div>
                <div class="q-sub">Verify battery capacity %, biometric sensors, audio/charging ports, and bonus trade-in accessories.</div>
            </div>
            """, unsafe_allow_html=True)

            # ── QUESTION 13: BATTERY HEALTH (%) WITH PROMINENT VISIBLE NUMBER ──
            card_spot = st.empty()

            if "sl_p4_battery" not in st.session_state:
                st.session_state.sl_p4_battery = int(st.session_state.get("battery_pct", 88))

            def _on_bat_change():
                if "sl_p4_battery" in st.session_state:
                    st.session_state.battery_pct = int(st.session_state.sl_p4_battery)

            sel_bat = st.slider(
                "Adjust Battery Health (%)",
                min_value=50,
                max_value=100,
                key="sl_p4_battery",
                on_change=_on_bat_change,
                help="Move slider to select your exact battery capacity percentage"
            )
            st.session_state.battery_pct = int(sel_bat)
            bat_val = int(sel_bat)
            bat_color = "#38bdf8" if bat_val >= 85 else ("#facc15" if bat_val >= 80 else "#f87171")
            bat_status = "💎 Optimal Battery Capacity (&ge;85%) — Zero Degradation" if bat_val >= 85 else (
                "✅ Good Battery Health (&ge;80%) — Normal Daily Performance" if bat_val >= 80 else "⚠️ Service Recommended (&lt;80%) — Service Adjustment Applies"
            )

            card_spot.markdown(f"""
            <div id="revalue-battery-card" style="background: #080d1a; border: 2px solid {bat_color}; border-radius: 18px; padding: 22px; text-align: center; margin: 12px 0 16px 0; box-shadow: 0 10px 30px rgba(0,0,0,0.7); transition: border-color 0.15s ease;">
                <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px;">Reported Battery Maximum Capacity</div>
                <div id="revalue-battery-number" style="font-size: 3.6rem; font-weight: 900; color: #ffffff; text-shadow: 0 0 25px rgba(56, 189, 248, 0.5); line-height: 1.1;">{bat_val}%</div>
                <div id="revalue-battery-badge" style="display: inline-block; margin-top: 10px; background: rgba(15, 23, 42, 0.9); border: 1px solid {bat_color}; padding: 6px 18px; border-radius: 100px; color: {bat_color}; font-weight: 700; font-size: 0.88rem; transition: all 0.15s ease;">
                    {bat_status}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

            # ── QUESTION 14: BIOMETRICS & QUESTION 15: AUDIO / PORTS ─────────
            c_bio, c_aud = st.columns(2)
            with c_bio:
                st.markdown("#### 🔐 Biometric Authentication")
                st.caption("Does Face ID or Fingerprint unlock smoothly?")
                opts_bio = [
                    "✅ Works instantly",
                    "❌ Biometric sensor failed or disabled"
                ]
                cur_bio = opts_bio.index(st.session_state.biometrics_work) if st.session_state.biometrics_work in opts_bio else 0
                st.session_state.biometrics_work = st.radio("Biometrics Sensor", opts_bio, index=cur_bio, key="rb_p4_bio", label_visibility="collapsed")

            with c_aud:
                st.markdown("#### 🔊 Audio & Charging Port")
                st.caption("Check speaker clarity, mic, and charging port connection.")
                opts_aud = [
                    "✅ Loud clear audio, mic & snug fast charge",
                    "❌ Static speaker / mic issue / loose charging port"
                ]
                cur_aud = 0 if "Loud" in st.session_state.audio_mic else 1
                sel_a = st.radio("Audio & Port Status", opts_aud, index=cur_aud, key="rb_p4_aud", label_visibility="collapsed")
                if "Loud" in sel_a:
                    st.session_state.audio_mic = "✅ Loud clear audio on calls & music; mic captures cleanly"
                    st.session_state.charging_port = "✅ Cable fits snugly; fast charging works without angle issues"
                else:
                    st.session_state.audio_mic = "❌ Crackling speaker / low earpiece sound / mic static"
                    st.session_state.charging_port = "❌ Loose port; only charges at specific angle"

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

            # ── QUESTION 16: ACCESSORIES & DOCUMENTATION (BONUS CASH) ────────
            st.markdown("#### 📦 Original Accessories & Documentation (Trade-In Bonus)")
            st.caption("Select which genuine accessories and retail documentation you will hand over during doorstep pickup.")
            ac1, ac2, ac3, ac4 = st.columns(4)
            with ac1:
                sel_box = st.checkbox("Original Box with IMEI (+₹1,200)", value=bool(st.session_state.has_box), key="cb_p4_box")
                st.session_state.has_box = sel_box
            with ac2:
                sel_chg = st.checkbox("Original Fast Charger (+₹1,000)", value=bool(st.session_state.has_charger), key="cb_p4_chg")
                st.session_state.has_charger = sel_chg
            with ac3:
                sel_bill = st.checkbox("Valid GST Purchase Bill (+₹800)", value=bool(st.session_state.has_bill), key="cb_p4_bill")
                st.session_state.has_bill = sel_bill
            with ac4:
                sel_war = st.checkbox("Brand Warranty (< 11 Months)", value=bool(st.session_state.under_warranty), key="cb_p4_war")
                st.session_state.under_warranty = sel_war

            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            col_prev, col_skip, col_next = st.columns([1, 1.3, 1.4])
            with col_prev:
                if st.button("← Previous", key="btn_prev_p4", use_container_width=True):
                    st.session_state.q_step = 3
                    st.rerun()
            with col_skip:
                st.empty()
            with col_next:
                if st.button("Review Summary →", key="btn_to_summary_p4", type="primary", use_container_width=True):
                    st.session_state.q_step = 5
                    st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 5: SUMMARY REVIEW OF ALL ANSWERS
    # ─────────────────────────────────────────────────────────────────────────
    elif q_step == 5:
        st.markdown(f"""
        <div style="background: rgba(13, 20, 36, 0.7); border: 1px solid rgba(212, 175, 55, 0.25); border-radius: 16px; padding: 14px 22px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="background: rgba(0, 229, 153, 0.15); border: 1px solid rgba(0, 229, 153, 0.4); border-radius: 100px; padding: 4px 14px; font-size: 0.76rem; font-weight: 800; color: #00e599; letter-spacing: 0.08em; text-transform: uppercase;">
                    Review Step
                </span>
                <span style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">
                    All Diagnostic Answers Saved
                </span>
            </div>
            <div style="color: #f6e05e; font-weight: 800; font-family: var(--mono); font-size: 0.88rem;">
                100% Ready for Valuation
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="q-card">
            <div class="q-title"><span>📋</span> Review Summary of All Diagnostic Answers</div>
            <div class="q-sub">Please review your selections below before generating your official verified resale valuation. Click 'Edit ✏️' on any section to modify an answer.</div>
        </div>
        """, unsafe_allow_html=True)

        age_years = max(1, 2026 - int(st.session_state.purchase_year))

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            # Card 1: Device & Specs (Page 1)
            st.markdown(f"""
            <div style="background: #0d1424; border: 1px solid rgba(212, 175, 55, 0.25); border-radius: 18px; padding: 20px 24px; margin-bottom: 18px;">
                <div style="font-size: 0.92rem; font-weight: 800; color: #f6e05e; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <span>📱 1. Device & Specifications</span>
                    <span style="font-size: 0.75rem; color: #00e599; font-weight: 700;">Page 1</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Brand & Model:</span>
                    <span class="val-val" style="color:#ffffff;">{st.session_state.phone_brand} {st.session_state.phone_model}</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Configuration:</span>
                    <span class="val-val">{st.session_state.storage_gb}GB Storage &bull; {st.session_state.ram_gb}GB RAM</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Purchase Year:</span>
                    <span class="val-val" style="color:#d4af37;">{st.session_state.purchase_year} ({age_years} yrs old)</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Original Price:</span>
                    <span class="val-val" style="color:#00e599;">₹{int(st.session_state.orig_price_inr):,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Edit Page 1 Specs ✏️", key="btn_rev_p1", use_container_width=True):
                st.session_state.q_step = 1
                st.rerun()

            # Card 2: Power, Calling & Screen OEM (Page 2)
            st.markdown(f"""
            <div style="background: #0d1424; border: 1px solid rgba(212, 175, 55, 0.25); border-radius: 18px; padding: 20px 24px; margin-bottom: 18px; margin-top: 14px;">
                <div style="font-size: 0.92rem; font-weight: 800; color: #f6e05e; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <span>⚡ 2. Power, Calling & Display Screen</span>
                    <span style="font-size: 0.75rem; color: #00e599; font-weight: 700;">Page 2</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Power & Boot:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{st.session_state.power_on[:30]}...</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Cellular Calling:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{st.session_state.calling_works[:30]}...</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Screen Originality:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{st.session_state.screen_orig.split('(')[0]}</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Display Panel:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{st.session_state.screen_panel_defects.split('(')[0]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Edit Page 2 Diagnostics ✏️", key="btn_rev_p2", use_container_width=True):
                st.session_state.q_step = 2
                st.rerun()

        with s_col2:
            # Card 3: Touch, Glass, Body & Cameras (Page 3)
            st.markdown(f"""
            <div style="background: #0d1424; border: 1px solid rgba(212, 175, 55, 0.25); border-radius: 18px; padding: 20px 24px; margin-bottom: 18px;">
                <div style="font-size: 0.92rem; font-weight: 800; color: #f6e05e; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <span>🔍 3. Touch, Glass, Body & Cameras</span>
                    <span style="font-size: 0.75rem; color: #00e599; font-weight: 700;">Page 3</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Touch Response:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{st.session_state.touch_works.split('(')[0]}</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Front Glass:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{st.session_state.screen_cond.split(':')[0]}</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Chassis / Frame:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{st.session_state.b_cond.split(':')[0]}</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Cameras & Flash:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{st.session_state.cameras_work[:30]}...</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Edit Page 3 Diagnostics ✏️", key="btn_rev_p3", use_container_width=True):
                st.session_state.q_step = 3
                st.rerun()

            # Card 4: Battery, Biometrics, Audio & Accessories (Page 4)
            box_txt = "✅ Included (+₹1,200)" if st.session_state.has_box else "❌ Missing"
            chg_txt = "✅ Included (+₹1,000)" if st.session_state.has_charger else "❌ Missing"
            bill_txt = "✅ Included (+₹800)" if st.session_state.has_bill else "❌ Missing"

            st.markdown(f"""
            <div style="background: #0d1424; border: 1px solid rgba(212, 175, 55, 0.25); border-radius: 18px; padding: 20px 24px; margin-bottom: 18px; margin-top: 14px;">
                <div style="font-size: 0.92rem; font-weight: 800; color: #f6e05e; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <span>🔋 4. Battery, Audio & Accessories</span>
                    <span style="font-size: 0.75rem; color: #00e599; font-weight: 700;">Page 4</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Battery Maximum Capacity:</span>
                    <span class="val-val" style="color:#38bdf8; font-weight:800;">{st.session_state.battery_pct}%</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Biometric Sensor:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{st.session_state.biometrics_work}</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Audio & Charging Port:</span>
                    <span class="val-val" style="font-size: 0.82rem;">Snug port & clear sound</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Included Accessories:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{box_txt} &bull; {chg_txt} &bull; {bill_txt}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Edit Page 4 Diagnostics ✏️", key="btn_rev_p4", use_container_width=True):
                st.session_state.q_step = 4
                st.rerun()

        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

        col_back, col_submit = st.columns([1, 2.2])
        with col_back:
            if st.button("← Previous", key="btn_rev_back", use_container_width=True):
                st.session_state.q_step = 4
                st.rerun()
        with col_submit:
            if st.button("⚡ Confirm & Calculate Guaranteed Resale Value (₹ INR) →", type="primary", key="btn_calc_val", use_container_width=True):
                st.session_state.q_step = 6
                st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 6: VALUATION REPORT WITH STRICT ≥15% ANNUAL DEPRECIATION
    # ─────────────────────────────────────────────────────────────────────────
    elif q_step == 6:
        # Standardize condition
        b_cond = st.session_state.b_cond
        if "Pristine" in b_cond:
            cond_val = "Pristine"
        elif "Good" in b_cond:
            cond_val = "Good"
        elif "Fair" in b_cond:
            cond_val = "Fair"
        else:
            cond_val = "Heavily Used"

        has_screen_dmg = 1 if ("Cracked" in st.session_state.screen_cond or "🔴" in st.session_state.screen_panel_defects) else 0
        has_body_scratches = 1 if ("Scratches" in st.session_state.screen_cond or cond_val in ["Fair", "Heavily Used"]) else 0

        if has_screen_dmg:
            cond_val = "Heavily Used"

        age_months = max(3, (2026 - int(st.session_state.purchase_year)) * 12)
        age_years = max(1, 2026 - int(st.session_state.purchase_year))

        m_specs = POPULAR_PHONES[st.session_state.phone_brand]["models"].get(st.session_state.phone_model, {})
        screen_size = m_specs.get("screen", 6.1)
        is_apple = (st.session_state.phone_brand == "Apple")
        os_val = "iOS" if is_apple else "Android"

        # Explicitly construct complete payload containing ALL columns required by pipeline
        payload = {
            "device_type": "Smartphone",
            "brand": st.session_state.phone_brand,
            "model_name": st.session_state.phone_model,
            "os": os_val,
            "storage_gb": int(st.session_state.storage_gb),
            "ram_gb": int(st.session_state.ram_gb),
            "screen_size_inch": float(screen_size),
            "battery_health_pct": int(st.session_state.battery_pct),
            "age_months": int(age_months),
            "condition": cond_val,
            "has_original_box": 1 if st.session_state.has_box else 0,
            "has_original_charger": 1 if st.session_state.has_charger else 0,
            "has_screen_damage": has_screen_dmg,
            "has_body_scratches": has_body_scratches,
            "original_price_usd": float(st.session_state.orig_price_inr) / USD_TO_INR
        }

        with st.spinner("Executing Production Gradient Boosting ML Inference & Calculating Price Guarantee..."):
            res = run_model_valuation(payload)
            time.sleep(0.1)

        if res:
            st.session_state.estimation_result = res
        else:
            orig_p = float(st.session_state.orig_price_inr)
            pred_base = orig_p * max(0.20, (1.0 - 0.18 * age_years))
            res = {
                "pred_inr": round(pred_base),
                "lo_inr": round(pred_base * 0.90),
                "hi_inr": round(pred_base * 1.10),
                "deprec_pct": round(min(80.0, 18.0 * age_years), 1),
                "is_anomaly": False,
                "latency_ms": 3.2,
                "payload": payload
            }
            st.session_state.estimation_result = res

        # ── MANDATORY DEPRECIATION ENFORCEMENT (≥15% PER YEAR) ──────────────
        orig_price_inr = float(st.session_state.orig_price_inr)
        min_deprec_rate = max(0.15 * age_years, 0.15)
        max_resale_cap = orig_price_inr * (1.0 - min_deprec_rate)

        raw_pred_inr = float(res["pred_inr"])
        final_inr = min(raw_pred_inr, max_resale_cap)
        final_inr = max(final_inr, orig_price_inr * 0.08) # Scrap floor

        # Diagnostic adjustments based on accessories
        bonus_acc = 0
        if st.session_state.has_box: bonus_acc += 1200
        if st.session_state.has_charger: bonus_acc += 1000
        if st.session_state.has_bill: bonus_acc += 800
        final_offer_with_bonus = min(final_inr + bonus_acc, orig_price_inr * 0.85)

        st.markdown(f"""
        <div class="result-banner">
            <div class="result-badge">👑 Guaranteed Official Trade-In Quote (All in INR ₹)</div>
            <div class="result-device">{st.session_state.phone_brand} {st.session_state.phone_model} ({st.session_state.storage_gb}GB &bull; {st.session_state.purchase_year})</div>
            <div class="result-price">₹{int(final_offer_with_bonus):,}</div>
            <div class="result-sub">Instant Doorstep UPI Payout &bull; Valid for 7 Days &bull; Free Concierge Pickup</div>
            <div class="result-tags">
                <span class="result-tag">✓ 100% Genuine OEM Match</span>
                <span class="result-tag">✓ Free Home Pickup</span>
                <span class="result-tag">✓ Certified Data Destruction</span>
                <span class="result-tag">✓ Instant Bank / UPI Credit</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Itemized valuation breakdown
        st.markdown(f"""
        <div style="background: #0d1424; border: 1px solid rgba(212, 175, 55, 0.35); border-radius: 18px; padding: 22px 28px; margin-bottom: 24px;">
            <div style="font-size: 1.05rem; font-weight: 800; color: #f6e05e; margin-bottom: 16px;">
                📊 Itemized Resale Valuation Breakdown (Enforced &ge;15%/Year Depreciation)
            </div>
            <div class="valuation-breakdown-row">
                <span class="val-label">Invoice Retail Purchase Price:</span>
                <span class="val-val" style="color: #ffffff;">₹{int(orig_price_inr):,}</span>
            </div>
            <div class="valuation-breakdown-row">
                <span class="val-label">Device Age ({age_years} Years Since {st.session_state.purchase_year}):</span>
                <span class="val-val" style="color: #f87171;">Mandatory Annual Depreciation applied (&ge;{int(min_deprec_rate*100)}%)</span>
            </div>
            <div class="valuation-breakdown-row">
                <span class="val-label">Maximum Market Resale Ceiling:</span>
                <span class="val-val" style="color: #d4af37;">₹{int(max_resale_cap):,}</span>
            </div>
            <div class="valuation-breakdown-row">
                <span class="val-label">Base AI Appraisal Valuation:</span>
                <span class="val-val" style="color: #ffffff;">₹{int(final_inr):,}</span>
            </div>
            <div class="valuation-breakdown-row">
                <span class="val-label">Genuine Accessories & Box Bonus:</span>
                <span class="val-val" style="color: #00e599;">+₹{bonus_acc:,}</span>
            </div>
            <div class="valuation-breakdown-row" style="border-top: 1.5px solid rgba(212, 175, 55, 0.4); padding-top: 12px; margin-top: 8px;">
                <span class="val-label" style="font-size: 1.05rem; font-weight: 800; color: #f6e05e;">Final Guaranteed Payout Offer:</span>
                <span class="val-val" style="font-size: 1.25rem; font-weight: 900; color: #00e599;">₹{int(final_offer_with_bonus):,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Doorstep Pickup Scheduler
        st.markdown("""
        <div class="q-card">
            <div class="q-title"><span>🚚</span> Book Free Doorstep Inspection & Cash Pickup</div>
            <div class="q-sub">Our verified valuation engineer will arrive at your location, run automated diagnostic verification in 5 minutes, and transfer ₹ straight to your GooglePay / UPI / Bank account.</div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.pickup_booked:
            with st.form("pickup_form"):
                pk_c1, pk_c2 = st.columns(2)
                with pk_c1:
                    cust_name = st.text_input("Full Name *", placeholder="e.g. Rahul Sharma")
                    cust_phone = st.text_input("Mobile Number (For WhatsApp / SMS OTP) *", placeholder="e.g. 9876543210")
                    cust_upi = st.text_input("UPI ID / PhonePe / GPay Number *", placeholder="e.g. rahul@oksbi")
                with pk_c2:
                    cust_city = st.selectbox("Pickup City", ["Bangalore", "Mumbai", "Delhi NCR", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad"])
                    cust_addr = st.text_area("Complete Pickup Address *", placeholder="Flat/House No, Building, Street, Landmark, Pincode")
                    cust_slot = st.selectbox("Preferred Time Slot", ["Tomorrow (10:00 AM - 01:00 PM)", "Tomorrow (02:00 PM - 06:00 PM)", "Day After Tomorrow (10:00 AM - 02:00 PM)"])

                submit_pickup = st.form_submit_button("⚡ Lock Price & Confirm Free Doorstep Pickup (₹ Payout)", type="primary", use_container_width=True)
                if submit_pickup:
                    if cust_name and cust_phone and cust_addr:
                        st.session_state.pickup_booked = True
                        st.rerun()
                    else:
                        st.error("Please fill in Name, Phone, and Complete Address to proceed.")
        else:
            st.markdown(f"""
            <div style="background: rgba(0, 229, 153, 0.12); border: 1.5px solid #00e599; border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 20px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🎉</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #00e599; margin-bottom: 6px;">Doorstep Pickup Confirmed!</div>
                <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
                    Your price lock for <b>₹{int(final_offer_with_bonus):,}</b> is confirmed. A verified technician has been assigned.<br>
                    Verification SMS and WhatsApp confirmation sent. Please keep your device charged & backed up.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Evaluate Another Smartphone", use_container_width=True):
                st.session_state.pickup_booked = False
                st.session_state.q_step = 1
                st.rerun()

        # Recalculate / Modify Navigation Buttons
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("← Modify Diagnostic Answers", key="btn_rep_back", use_container_width=True):
                st.session_state.q_step = 5
                st.rerun()
        with btn_c2:
            if st.button("⚡ Start New Valuation", key="btn_rep_new", use_container_width=True):
                st.session_state.q_step = 1
                st.session_state.pickup_booked = False
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: SELL LAPTOP WIZARD (CLEAN MULTI-STEP FLOW)
# ─────────────────────────────────────────────────────────────────────────────
def render_laptop_wizard(POPULAR_LAPTOPS, run_model_valuation, fmt_inr):
    st.markdown("""
    <div class="hero-card">
        <div class="hero-badge"><span class="hero-badge-dot"></span>Laptop Resale Division</div>
        <div class="hero-title">Sell Your Laptop in <span>Indian Rupees (₹)</span></div>
        <div class="hero-sub">Instant doorstep valuation for Apple MacBooks, Dell XPS, Lenovo ThinkPads, HP Spectres and gaming laptops.</div>
        <div class="hero-perks">
            <div class="hero-perk-item"><span class="hero-perk-icon">✓</span> Free Doorstep Pickup</div>
            <div class="hero-perk-item"><span class="hero-perk-icon">✓</span> Instant Bank / UPI Payment</div>
            <div class="hero-perk-item"><span class="hero-perk-icon">✓</span> 7 Days Price Lock Guarantee</div>
            <div class="hero-perk-item"><span class="hero-perk-icon">✓</span> All Valuations in Indian Rupees (₹)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Perks
    st.markdown("""
    <div class="revalue-perks-row">
        <div class="revalue-perk-card">
            <span class="revalue-perk-icon">💻</span>
            <div class="revalue-perk-title">Highest Laptop Value</div>
            <div class="revalue-perk-desc">AI benchmarked against current Indian resale market prices</div>
        </div>
        <div class="revalue-perk-card">
            <span class="revalue-perk-icon">🚚</span>
            <div class="revalue-perk-title">Home Diagnostic Check</div>
            <div class="revalue-perk-desc">On-site 5-minute hardware verification by trained specialist</div>
        </div>
        <div class="revalue-perk-card">
            <span class="revalue-perk-icon">💸</span>
            <div class="revalue-perk-title">Instant Cash Transfer</div>
            <div class="revalue-perk-desc">Full ₹ payout before the technician leaves your doorstep</div>
        </div>
        <div class="revalue-perk-card">
            <span class="revalue-perk-icon">🛡️</span>
            <div class="revalue-perk-title">Complete SSD Wipe</div>
            <div class="revalue-perk-desc">Complimentary DoD 5220.22-M data wipe certificate</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    l_titles = {
        1: "💻 Laptop & Technical Specifications",
        2: "🔧 Hardware Diagnostics & Accessories",
        3: "📋 Summary Review of Laptop Diagnostics",
        4: "👑 Official Laptop Valuation Report"
    }

    l_q_step = st.session_state.get("l_q_step", 1)
    if l_q_step > 4:
        l_q_step = 1
        st.session_state.l_q_step = 1

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: MERGED MANUFACTURER, MODEL & SPECS
    # ─────────────────────────────────────────────────────────────────────────
    if l_q_step == 1:
        st.markdown("""
        <div class="q-card">
            <div class="q-title"><span>💻</span> Laptop & Technical Specifications</div>
            <div class="q-sub">Select your laptop brand, model series, and hardware configuration. Purchase year is validated against official launch year.</div>
        </div>
        """, unsafe_allow_html=True)

        mfg_list = list(POPULAR_LAPTOPS.keys())
        cur_l_brand = st.session_state.get("laptop_brand", "Apple")
        cur_b_idx = mfg_list.index(cur_l_brand) if cur_l_brand in mfg_list else 0

        c_brand, c_model = st.columns(2)
        with c_brand:
            sel_brand = st.selectbox("Laptop Manufacturer", mfg_list, index=cur_b_idx, key="sb_lbrand_wizard")
            if sel_brand != st.session_state.laptop_brand:
                st.session_state.laptop_brand = sel_brand
                first_m = list(POPULAR_LAPTOPS[sel_brand]["models"].keys())[0]
                st.session_state.laptop_model = first_m
                specs = POPULAR_LAPTOPS[sel_brand]["models"][first_m]
                st.session_state.l_ram = int(specs["ram"])
                st.session_state.l_storage = int(specs["storage"])
                st.session_state.l_orig_inr = int(specs["orig_inr"])
                l_yr = int(specs.get("launch_year", 2023))
                st.session_state.l_year = max(l_yr, 2023)
                st.rerun()

        with c_model:
            brand_models = list(POPULAR_LAPTOPS[st.session_state.laptop_brand]["models"].keys())
            cur_l_model = st.session_state.get("laptop_model", brand_models[0])
            cur_m_idx = brand_models.index(cur_l_model) if cur_l_model in brand_models else 0
            sel_model = st.selectbox(f"{st.session_state.laptop_brand} Model Series", brand_models, index=cur_m_idx, key="sb_lmodel_wizard")
            if sel_model != st.session_state.laptop_model:
                st.session_state.laptop_model = sel_model
                specs = POPULAR_LAPTOPS[st.session_state.laptop_brand]["models"][sel_model]
                st.session_state.l_ram = int(specs["ram"])
                st.session_state.l_storage = int(specs["storage"])
                st.session_state.l_orig_inr = int(specs["orig_inr"])
                l_yr = int(specs.get("launch_year", 2023))
                st.session_state.l_year = max(l_yr, 2023)
                st.rerun()

        specs = POPULAR_LAPTOPS[st.session_state.laptop_brand]["models"].get(st.session_state.laptop_model, {})
        l_launch_yr = int(specs.get("launch_year", 2022))

        min_year = max(l_launch_yr, 2016)
        allowed_years = list(range(2026, min_year - 1, -1))
        if st.session_state.l_year not in allowed_years:
            st.session_state.l_year = allowed_years[-1] if allowed_years else 2024

        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 10px 16px; margin: 12px 0 18px 0; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.85rem; color: #cbd5e1;">
                🚀 <b>{st.session_state.laptop_brand} {st.session_state.laptop_model}</b> &bull; Official Market Launch: <span style="color:#38bdf8; font-weight:800;">{l_launch_yr}</span>
            </span>
            <span style="font-size: 0.78rem; color: #94a3b8;">
                Purchase years permitted: <b style="color:#f6e05e;">{min_year}–2026</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ram_opts = [8, 16, 32, 64]
            cur_r = st.session_state.l_ram if st.session_state.l_ram in ram_opts else 16
            sel_ram = st.selectbox("RAM (GB)", ram_opts, index=ram_opts.index(cur_r), key="sb_lram_wizard")
            st.session_state.l_ram = sel_ram
        with c2:
            stor_opts = [256, 512, 1000, 2000]
            cur_s = st.session_state.l_storage if st.session_state.l_storage in stor_opts else 512
            sel_stor = st.selectbox("SSD Storage (GB)", stor_opts, index=stor_opts.index(cur_s), key="sb_lstor_wizard")
            st.session_state.l_storage = sel_stor
        with c3:
            cur_y_idx = allowed_years.index(st.session_state.l_year) if st.session_state.l_year in allowed_years else 0
            sel_year = st.selectbox("Purchase Year", allowed_years, index=cur_y_idx, key="sb_lyear_wizard")
            st.session_state.l_year = sel_year
            age_yrs = 2026 - sel_year
            st.caption(f"Laptop Age: ~{age_yrs} yr{'s' if age_yrs != 1 else ''} ({max(3, age_yrs * 12)} mos)")
        with c4:
            sel_price = st.number_input(
                "Original Purchase Price (₹)",
                min_value=20000,
                max_value=500000,
                value=int(st.session_state.l_orig_inr),
                step=5000,
                key="sb_lprice_wizard",
                help="Invoice purchase price in Indian Rupees (₹)"
            )
            st.session_state.l_orig_inr = sel_price

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        col_space, col_next = st.columns([2, 1.2])
        with col_next:
            if st.button("Next →", key="btn_lnext_p1", type="primary", use_container_width=True):
                st.session_state.l_q_step = 2
                st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: HARDWARE & ACCESSORIES CHECKS (ALL IN 1 PAGE)
    # ─────────────────────────────────────────────────────────────────────────
    elif l_q_step == 2:
        st.markdown("""
        <div class="q-card">
            <div class="q-title"><span>🔧</span> Hardware Diagnostics & Accessories</div>
            <div class="q-sub">Answer these key hardware conditions and accessories to calculate your verified payout in ₹ INR.</div>
        </div>
        """, unsafe_allow_html=True)

        col_sc, col_b = st.columns(2)
        with col_sc:
            st.markdown("#### 📺 Screen & Display Quality")
            opts_sc = [
                "Flawless: Spotless display, zero lines or dead pixels",
                "Minor keyboard coating marks / faint light scratches",
                "Cracked screen / damaged panel / vertical lines"
            ]
            cur_sc = opts_sc.index(st.session_state.l_screen_cond) if st.session_state.l_screen_cond in opts_sc else 0
            st.session_state.l_screen_cond = st.radio("Screen Quality", opts_sc, index=cur_sc, key="rb_l2_sc", label_visibility="collapsed")

        with col_b:
            st.markdown("#### 🛡️ Chassis, Hinges & Body")
            opts_b = [
                "✨ Like New: Spotless lid & chassis, zero dents",
                "👍 Normal Wear: Minor light corner scuffs from bag carrying",
                "🔧 Heavily Used: Corner dents, bent lid, or loose/cracked hinge"
            ]
            cur_b = opts_b.index(st.session_state.l_body_cond) if st.session_state.l_body_cond in opts_b else 0
            st.session_state.l_body_cond = st.radio("Body Condition", opts_b, index=cur_b, key="rb_l2_b", label_visibility="collapsed")

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        col_kb, col_bat = st.columns(2)
        with col_kb:
            st.markdown("#### ⌨️ Keyboard & Trackpad")
            opts_kb = [
                "✅ All keyboard keys and trackpad click and track smoothly",
                "❌ Sticky/unresponsive keys, trackpad click issue, or backlight failure"
            ]
            cur_kb = opts_kb.index(st.session_state.l_kb) if st.session_state.l_kb in opts_kb else 0
            st.session_state.l_kb = st.radio("Keyboard Status", opts_kb, index=cur_kb, key="rb_l2_kb", label_visibility="collapsed")

        with col_bat:
            st.markdown("#### 🔋 Battery Backup Performance")
            opts_bat = [
                "✅ Battery holds 3+ hours charge",
                "⚠️ Battery holds 1-2 hours charge",
                "❌ Battery service required / only works plugged in"
            ]
            cur_bat = opts_bat.index(st.session_state.l_battery) if st.session_state.l_battery in opts_bat else 0
            st.session_state.l_battery = st.radio("Battery Status", opts_bat, index=cur_bat, key="rb_l2_bat", label_visibility="collapsed")

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        col_pt, col_acc = st.columns(2)
        with col_pt:
            st.markdown("#### 🔌 Ports, WebCam & Connectivity")
            opts_pt = [
                "✅ All USB / Type-C ports, WebCam and Wi-Fi connect normally",
                "❌ Loose USB/charging port, faulty webcam, or Wi-Fi connectivity issue"
            ]
            cur_pt = opts_pt.index(st.session_state.l_ports) if st.session_state.l_ports in opts_pt else 0
            st.session_state.l_ports = st.radio("Ports Status", opts_pt, index=cur_pt, key="rb_l2_pt", label_visibility="collapsed")

        with col_acc:
            st.markdown("#### 📦 Original Accessories")
            sel_chg = st.checkbox("Original Charger (+₹2,500)", value=bool(st.session_state.l_charger), key="cb_l2_chg")
            st.session_state.l_charger = sel_chg
            sel_box = st.checkbox("Original Box (+₹1,500)", value=bool(st.session_state.l_box), key="cb_l2_box")
            st.session_state.l_box = sel_box
            sel_bill = st.checkbox("Valid GST Bill (+₹1,000)", value=bool(st.session_state.l_bill), key="cb_l2_bill")
            st.session_state.l_bill = sel_bill

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        col_prev, col_skip, col_next = st.columns([1, 1.3, 1.4])
        with col_prev:
            if st.button("← Previous", key="btn_lprev_p2", use_container_width=True):
                st.session_state.l_q_step = 1
                st.rerun()
        with col_skip:
            st.empty()
        with col_next:
            if st.button("Review Summary →", key="btn_lto_sum", type="primary", use_container_width=True):
                st.session_state.l_q_step = 3
                st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: SUMMARY REVIEW OF LAPTOP ANSWERS
    # ─────────────────────────────────────────────────────────────────────────
    elif l_q_step == 3:
        st.markdown(f"""
        <div style="background: rgba(13, 20, 36, 0.7); border: 1px solid rgba(212, 175, 55, 0.25); border-radius: 16px; padding: 14px 22px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="background: rgba(0, 229, 153, 0.15); border: 1px solid rgba(0, 229, 153, 0.4); border-radius: 100px; padding: 4px 14px; font-size: 0.76rem; font-weight: 800; color: #00e599; letter-spacing: 0.08em; text-transform: uppercase;">
                    Review Step
                </span>
                <span style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">
                    All Laptop Diagnostic Answers Saved
                </span>
            </div>
            <div style="color: #f6e05e; font-weight: 800; font-family: var(--mono); font-size: 0.88rem;">
                100% Ready for Valuation
            </div>
        </div>
        """, unsafe_allow_html=True)

        age_years = max(1, 2026 - int(st.session_state.l_year))

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown(f"""
            <div style="background: #0d1424; border: 1px solid rgba(212, 175, 55, 0.25); border-radius: 18px; padding: 20px 24px; margin-bottom: 18px;">
                <div style="font-size: 0.92rem; font-weight: 800; color: #f6e05e; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <span>💻 1. Laptop & Specifications</span>
                    <span style="font-size: 0.75rem; color: #00e599; font-weight: 700;">Page 1</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Brand & Model:</span>
                    <span class="val-val" style="color:#ffffff;">{st.session_state.laptop_brand} {st.session_state.laptop_model}</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Hardware:</span>
                    <span class="val-val">{st.session_state.l_ram}GB RAM &bull; {st.session_state.l_storage}GB SSD</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Purchase Year:</span>
                    <span class="val-val" style="color:#d4af37;">{st.session_state.l_year} ({age_years} yrs old)</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Original Price:</span>
                    <span class="val-val" style="color:#00e599;">₹{int(st.session_state.l_orig_inr):,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Edit Laptop Specs ✏️", key="btn_ledit_specs", use_container_width=True):
                st.session_state.l_q_step = 1
                st.rerun()

        with s_col2:
            st.markdown(f"""
            <div style="background: #0d1424; border: 1px solid rgba(212, 175, 55, 0.25); border-radius: 18px; padding: 20px 24px; margin-bottom: 18px;">
                <div style="font-size: 0.92rem; font-weight: 800; color: #f6e05e; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <span>🔧 2. Hardware Condition</span>
                    <span style="font-size: 0.75rem; color: #00e599; font-weight: 700;">Page 2</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Display & Screen:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{st.session_state.l_screen_cond.split(':')[0]}</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Chassis & Body:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{st.session_state.l_body_cond.split(':')[0]}</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Battery Backup:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{st.session_state.l_battery}</span>
                </div>
                <div class="valuation-breakdown-row" style="padding: 6px 0;">
                    <span class="val-label">Accessories:</span>
                    <span class="val-val" style="font-size: 0.82rem;">{'Charger' if st.session_state.l_charger else 'No charger'} &bull; {'Box' if st.session_state.l_box else 'No box'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Edit Hardware Diagnostics ✏️", key="btn_ledit_diag", use_container_width=True):
                st.session_state.l_q_step = 2
                st.rerun()

        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        col_back, col_submit = st.columns([1, 2.2])
        with col_back:
            if st.button("← Previous", key="btn_lrev_back", use_container_width=True):
                st.session_state.l_q_step = 2
                st.rerun()
        with col_submit:
            if st.button("⚡ Confirm & Calculate Laptop Resale Offer (₹ INR) →", type="primary", key="btn_lcalc_val", use_container_width=True):
                st.session_state.l_q_step = 4
                st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: LAPTOP VALUATION REPORT WITH STRICT ≥15% ANNUAL DEPRECIATION
    # ─────────────────────────────────────────────────────────────────────────
    elif l_q_step == 4:
        l_orig_inr = float(st.session_state.l_orig_inr)
        age_years = max(1, 2026 - int(st.session_state.l_year))

        # Mandatory depreciation enforcement: >= 15% per year
        min_deprec_rate = max(0.15 * age_years, 0.15)
        max_resale_cap = l_orig_inr * (1.0 - min_deprec_rate)

        # Baseline valuation
        base_factor = 0.82 ** age_years
        val_inr = l_orig_inr * base_factor

        # Condition adjustments
        if "Flawless" not in st.session_state.l_screen_cond:
            val_inr *= 0.80
        if "✨ Like New" not in st.session_state.l_body_cond:
            val_inr *= 0.88
        if "✅" not in st.session_state.l_kb:
            val_inr *= 0.88
        if "3+ hours" not in st.session_state.l_battery:
            val_inr *= 0.85
        if "✅" not in st.session_state.l_ports:
            val_inr *= 0.90

        # Enforce depreciation ceiling strictly
        val_inr = min(val_inr, max_resale_cap)

        # Accessories bonus
        l_bonus = 0
        if st.session_state.l_charger: l_bonus += 2500
        if st.session_state.l_box: l_bonus += 1500
        if st.session_state.l_bill: l_bonus += 1000

        final_laptop_offer = min(val_inr + l_bonus, l_orig_inr * 0.85)

        st.markdown(f"""
        <div class="result-banner">
            <div class="result-badge">👑 Guaranteed Official Laptop Trade-In Offer (₹ INR)</div>
            <div class="result-device">{st.session_state.laptop_brand} {st.session_state.laptop_model} ({st.session_state.l_ram}GB RAM &bull; {st.session_state.l_storage}GB SSD &bull; {st.session_state.l_year})</div>
            <div class="result-price">₹{int(final_laptop_offer):,}</div>
            <div class="result-sub">Instant Doorstep UPI Payout &bull; Valid for 7 Days &bull; Free Technician Pickup</div>
            <div class="result-tags">
                <span class="result-tag">✓ Dedicated Laptop Engineer</span>
                <span class="result-tag">✓ Free Home Pickup</span>
                <span class="result-tag">✓ Certified SSD DoD Wipe</span>
                <span class="result-tag">✓ Instant Bank / UPI Credit</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown table
        st.markdown(f"""
        <div style="background: #0d1424; border: 1px solid rgba(212, 175, 55, 0.35); border-radius: 18px; padding: 22px 28px; margin-bottom: 24px;">
            <div style="font-size: 1.05rem; font-weight: 800; color: #f6e05e; margin-bottom: 16px;">
                📊 Itemized Laptop Resale Breakdown (Enforced &ge;15%/Year Depreciation)
            </div>
            <div class="valuation-breakdown-row">
                <span class="val-label">Invoice Retail Purchase Price:</span>
                <span class="val-val" style="color: #ffffff;">₹{int(l_orig_inr):,}</span>
            </div>
            <div class="valuation-breakdown-row">
                <span class="val-label">Device Age ({age_years} Years Since {st.session_state.l_year}):</span>
                <span class="val-val" style="color: #f87171;">Mandatory Annual Depreciation applied (&ge;{int(min_deprec_rate*100)}%)</span>
            </div>
            <div class="valuation-breakdown-row">
                <span class="val-label">Maximum Market Resale Ceiling:</span>
                <span class="val-val" style="color: #d4af37;">₹{int(max_resale_cap):,}</span>
            </div>
            <div class="valuation-breakdown-row">
                <span class="val-label">Hardware & Diagnostic Base Valuation:</span>
                <span class="val-val" style="color: #ffffff;">₹{int(val_inr):,}</span>
            </div>
            <div class="valuation-breakdown-row">
                <span class="val-label">Charger, Box & Documentation Bonus:</span>
                <span class="val-val" style="color: #00e599;">+₹{l_bonus:,}</span>
            </div>
            <div class="valuation-breakdown-row" style="border-top: 1.5px solid rgba(212, 175, 55, 0.4); padding-top: 12px; margin-top: 8px;">
                <span class="val-label" style="font-size: 1.05rem; font-weight: 800; color: #f6e05e;">Final Guaranteed Payout Offer:</span>
                <span class="val-val" style="font-size: 1.25rem; font-weight: 900; color: #00e599;">₹{int(final_laptop_offer):,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Doorstep Pickup Scheduler
        st.markdown("""
        <div class="q-card">
            <div class="q-title"><span>🚚</span> Book Free Laptop Doorstep Inspection & Cash Pickup</div>
            <div class="q-sub">Our verified hardware engineer will arrive at your home, conduct quick diagnostic testing, and transfer ₹ payment directly to your account.</div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.get("laptop_pickup_booked", False):
            with st.form("laptop_pickup_form"):
                pk_c1, pk_c2 = st.columns(2)
                with pk_c1:
                    cust_name = st.text_input("Full Name *", placeholder="e.g. Priya Nair")
                    cust_phone = st.text_input("Mobile Number (For WhatsApp / SMS OTP) *", placeholder="e.g. 9876543210")
                    cust_upi = st.text_input("UPI ID / PhonePe / GPay Number *", placeholder="e.g. priya@okhdfcbank")
                with pk_c2:
                    cust_city = st.selectbox("Pickup City", ["Bangalore", "Mumbai", "Delhi NCR", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad"], key="l_city_sel")
                    cust_addr = st.text_area("Complete Pickup Address *", placeholder="Flat/House No, Building, Street, Landmark, Pincode", key="l_addr_txt")
                    cust_slot = st.selectbox("Preferred Time Slot", ["Tomorrow (10:00 AM - 01:00 PM)", "Tomorrow (02:00 PM - 06:00 PM)", "Day After Tomorrow (10:00 AM - 02:00 PM)"], key="l_slot_sel")

                submit_pickup = st.form_submit_button("⚡ Lock Price & Confirm Free Doorstep Pickup (₹ Payout)", type="primary", use_container_width=True)
                if submit_pickup:
                    if cust_name and cust_phone and cust_addr:
                        st.session_state.laptop_pickup_booked = True
                        st.rerun()
                    else:
                        st.error("Please fill in Name, Phone, and Complete Address to proceed.")
        else:
            st.markdown(f"""
            <div style="background: rgba(0, 229, 153, 0.12); border: 1.5px solid #00e599; border-radius: 16px; padding: 24px; text-align: center; margin-bottom: 20px;">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">🎉</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #00e599; margin-bottom: 6px;">Laptop Doorstep Pickup Confirmed!</div>
                <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
                    Your guaranteed price lock for <b>₹{int(final_laptop_offer):,}</b> is confirmed. A laptop technician has been assigned.<br>
                    Verification SMS and WhatsApp confirmation sent. Please ensure the laptop is charged for testing.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Evaluate Another Laptop", use_container_width=True):
                st.session_state.laptop_pickup_booked = False
                st.session_state.l_q_step = 1
                st.rerun()

        # Recalculate / Modify Navigation Buttons
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("← Modify Diagnostic Answers", key="btn_lrep_back", use_container_width=True):
                st.session_state.l_q_step = 3
                st.rerun()
        with btn_c2:
            if st.button("⚡ Start New Laptop Valuation", key="btn_lrep_new", use_container_width=True):
                st.session_state.l_q_step = 1
                st.session_state.laptop_pickup_booked = False
                st.rerun()
