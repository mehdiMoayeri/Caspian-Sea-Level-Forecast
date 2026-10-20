import os
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ==============================================================================
# ۱. تنظیمات صفحه، فونت فارسی وزیرمتن (Vazirmatn) و پوسته گرافیکی حرفه‌ای RTL
# ==============================================================================
st.set_page_config(
    page_title="سامانه پایش و پیش‌بینی ۱۲ ماهه تراز دریای خزر",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&display=swap');

    /* اعمال فونت وزیرمتن روی تمامی اجزای سامانه */
    html, body, [class*="css"], .stMarkdown, .stText, .stButton button, .stSelectbox, .stSlider {
        font-family: 'Vazirmatn', 'Tahoma', sans-serif !important;
    }

    /* راست‌به‌چپ سازی متون و باکس‌های فارسی */
    .main-header {
        background: linear-gradient(135deg, #071930 0%, #1e3a8a 60%, #1d4ed8 100%);
        color: white;
        padding: 1.4rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        direction: rtl;
        text-align: right;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.15);
        border-right: 6px solid #38bdf8;
    }
    .main-header h1 {
        color: #ffffff !important;
        font-family: 'Vazirmatn', sans-serif !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        margin-bottom: 0.45rem;
        line-height: 1.5;
    }
    .main-header p {
        color: #e0f2fe;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
        line-height: 1.6;
    }

    /* باکس‌های راهنما و توضیحات تب‌ها */
    .rtl-box {
        direction: rtl;
        text-align: right;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-right: 5px solid #2563eb;
        padding: 1.1rem 1.4rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
        font-size: 0.95rem;
        line-height: 1.85;
        color: #1e293b;
    }

    /* طراحی کارت‌های شاخص (KPI Cards) بدون بریده شدن متن */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 4px solid #1d4ed8;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        direction: rtl;
        text-align: right;
        height: 100%;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        direction: ltr;
        text-align: right;
        margin-bottom: 6px;
    }
    .kpi-delta {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        background-color: #f1f5f9;
        color: #334155;
    }

    /* استایل برجسته و دکمه‌ای برای تب‌های ۵گانه */
    div[data-baseweb="tab-list"] {
        direction: rtl;
        gap: 10px;
        background-color: #f1f5f9;
        padding: 8px 10px;
        border-radius: 12px;
        border: 1px solid #cbd5e1;
        margin-top: 1rem;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    button[data-baseweb="tab"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 9px !important;
        padding: 10px 18px !important;
        font-family: 'Vazirmatn', sans-serif !important;
        font-size: 0.96rem !important;
        font-weight: 700 !important;
        color: #334155 !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
    }
    button[data-baseweb="tab"]:hover {
        border-color: #2563eb !important;
        color: #1d4ed8 !important;
        background-color: #eff6ff !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border-color: #1e3a8a !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.28) !important;
    }

    /* راست‌چین کردن عناوین بخش‌ها و نوار کناری */
    h2, h3, h4, h5, h6 {
        direction: rtl;
        text-align: right;
        font-family: 'Vazirmatn', sans-serif !important;
        font-weight: 700 !important;
        color: #0f172a;
    }
    section[data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
        background-color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# هدر اصلی سامانه
st.markdown(
    """
    <div class="main-header">
        <h1>🌊 سامانه عملیاتی پایش و پیش‌بینی احتمالاتی ۱۲ ماهه تراز آب دریای خزر (فاز دوم)</h1>
        <p>مرکز مطالعات و تحقیقات دریای خزر | ترکیب هوشمند سامانه دینامیکی GloFAS v4.0 (افق ۱ تا ۶ ماه)، شبکه عصبی عمیق LSTM (افق ۱ تا ۱۲ ماه) و مدل بیلان آب دریای خزر</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# ۲. بارگذاری هوشمند بسته داده (caspian_streamlit_bundle.pkl)
# ==============================================================================
@st.cache_data
def load_bundle_from_path(filepath: str):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    for k in ["historical_master", "glofas_seasonal_6m", "dl_forecast_12m", "ensemble_inflow_12m", "csl_forecast_12m"]:
        if k in data and "Date" in data[k].columns:
            data[k]["Date"] = pd.to_datetime(data[k]["Date"])
    return data

default_paths = [
    Path("./caspian_streamlit_bundle.pkl"),
    Path("./Caspian_Phase2_Database/08_API_JSON_GeoJSON_NetCDF/caspian_streamlit_bundle.pkl"),
    Path("../Caspian_Phase2_Database/08_API_JSON_GeoJSON_NetCDF/caspian_streamlit_bundle.pkl"),
]

bundle = None
for p in default_paths:
    if p.is_file():
        bundle = load_bundle_from_path(str(p))
        break

with st.sidebar:
    st.markdown("### ⚙️ تنظیمات سامانه و سناریوها")
    if bundle is None:
        st.warning("فایل `caspian_streamlit_bundle.pkl` یافت نشد. لطفاً آن را بارگذاری کنید:")
        uploaded_pkl = st.file_uploader("انتخاب فایل pkl", type=["pkl"])
        if uploaded_pkl is not None:
            bundle = pickle.load(uploaded_pkl)
            for k in ["historical_master", "glofas_seasonal_6m", "dl_forecast_12m", "ensemble_inflow_12m", "csl_forecast_12m"]:
                if k in bundle and "Date" in bundle[k].columns:
                    bundle[k]["Date"] = pd.to_datetime(bundle[k]["Date"])
    else:
        st.success("✅ متصل به پایگاه داده یکپارچه فاز دوم")

if bundle is None:
    st.info("لطفاً فایل `caspian_streamlit_bundle.pkl` را کنار فایل `app.py` قرار دهید.")
    st.stop()

# استخراج جداول از بسته داده
df_outlets  = bundle["outlets_meta"]
df_master   = bundle["historical_master"]
df_clim     = bundle["climatology_12m"]
df_seas_6m  = bundle["glofas_seasonal_6m"]
df_dl_12m   = bundle["dl_forecast_12m"]
df_ens_12m  = bundle["ensemble_inflow_12m"]
df_csl_12m  = bundle["csl_forecast_12m"].copy()
df_skill    = bundle["hindcast_skill"]
df_lag_corr = bundle["lag_correlations"]
df_hypso    = bundle["hypsometry"].sort_values("Elevation_m_BS").reset_index(drop=True)
csl_mc_base = bundle["csl_mc_paths"]

initial_csl_m = float(df_master.iloc[-1]["CSL_Satellite_Altimetry_m"])

# ==============================================================================
# ۳. کنترلرهای تعاملی نوار کناری
# ==============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("#### 🚨 ۱. تنظیم آستانه‌های هشدار و بحران")
    warn_thresh = st.slider(
        "آستانه هشدار زرد (متر بالتیک):",
        min_value=-29.60, max_value=-28.80, value=-29.30, step=0.01, format="%.2f"
    )
    crit_thresh = st.slider(
        "آستانه بحرانی قرمز (متر بالتیک):",
        min_value=-29.70, max_value=-28.90, value=-29.38, step=0.01, format="%.2f"
    )

    st.markdown("---")
    st.markdown("#### 🎛️ ۲. شبیه‌ساز سناریوهای مدیریتی")
    st.caption("بررسی اثر تغییر ورودی‌ها بر تراز ۱۲ ماه آینده دریای خزر:")
    inflow_pct_change = st.slider(
        "تغییر دبی ورودی رودخانه‌ها (ولگا و غیره) [%]:",
        min_value=-40, max_value=40, value=0, step=5
    )
    evap_pct_change = st.slider(
        "تغییر نرخ تبخیر از سطح دریا [%]:",
        min_value=-25, max_value=25, value=0, step=5
    )
    karabogaz_open_pct = st.slider(
        "درصد باز بودن خروجی تنگه قره‌بوغاز [%]:",
        min_value=0, max_value=120, value=100, step=10
    )
    hist_start_year = st.selectbox(
        "شروع نمایش سری تاریخی تراز در نمودار:",
        options=[2023, 2020, 2015, 2000, 1979], index=0
    )

# ==============================================================================
# ۴. محاسبه زنده سناریوی تعاملی (What-If)
# ==============================================================================
def get_area_km2(h_m):
    return float(np.interp(h_m, df_hypso["Elevation_m_BS"], df_hypso["Area_1000_km2"]) * 1000.0)

scenario_delta_h = np.zeros(len(df_csl_12m))
cum_dh_scen = 0.0
for idx, row in df_csl_12m.iterrows():
    h_ref = row["CSL_P50_m (Median_Forecast)"]
    area_k = get_area_km2(h_ref)
    dq_vol  = row["Inflow_Vol_km3"] * (inflow_pct_change / 100.0)
    m_num = row["Date"].month
    clim_evap_mm = float(df_clim.loc[df_clim["Month"] == m_num, "Clim_Evap_mm"].values[0]) * 1.56
    evap_vol_base = (clim_evap_mm / 1e6) * area_k
    de_vol  = -evap_vol_base * (evap_pct_change / 100.0)
    dkb_vol = -row["KaraBogaz_Outflow_km3"] * ((karabogaz_open_pct - 100.0) / 100.0)
    dV_scen = dq_vol + de_vol + dkb_vol
    cum_dh_scen += (dV_scen / area_k) * 1000.0
    scenario_delta_h[idx] = cum_dh_scen

csl_mc_adj = csl_mc_base + scenario_delta_h[np.newaxis, :]
df_csl_12m["Scenario_P50_m"] = np.round(df_csl_12m["CSL_P50_m (Median_Forecast)"] + scenario_delta_h, 3)
df_csl_12m["Scenario_P10_m"] = np.round(df_csl_12m["CSL_P10_m (Low_Level_Risk)"] + scenario_delta_h, 3)
df_csl_12m["Scenario_P90_m"] = np.round(df_csl_12m["CSL_P90_m (High_Level)"] + scenario_delta_h, 3)
df_csl_12m["Dynamic_Risk_Warn (%)"] = [round(float(np.mean(csl_mc_adj[:, i] < warn_thresh) * 100.0), 1) for i in range(12)]
df_csl_12m["Dynamic_Risk_Crit (%)"] = [round(float(np.mean(csl_mc_adj[:, i] < crit_thresh) * 100.0), 1) for i in range(12)]

# ==============================================================================
# ۵. کارت‌های شاخص کلیدی عملکرد (با طراحی سفارشی RTL بدون بریده شدن متن)
# ==============================================================================
m6_row  = df_csl_12m.iloc[5]
m12_row = df_csl_12m.iloc[11]
tot_12m_inflow_km3 = float(df_csl_12m["Inflow_Vol_km3"].sum()) * (1.0 + inflow_pct_change / 100.0)
max_crit_risk_pct  = float(df_csl_12m["Dynamic_Risk_Crit (%)"].max())
diff_6m_cm  = (m6_row["Scenario_P50_m"] - initial_csl_m) * 100.0
diff_12m_cm = (m12_row["Scenario_P50_m"] - initial_csl_m) * 100.0

kpi_cols = st.columns(5)
cards_data = [
    ("📍 تراز اولیه دریا (سپتامبر ۲۰۲۶)", f"{initial_csl_m:.3f} m", "مبنای ارتفاعی بالتیک (BS)", "#e2e8f0", "#334155"),
    ("🔮 پیش‌بینی افق ۶ ماهه (P50 - مارس ۲۰۲۷)", f"{m6_row['Scenario_P50_m']:.3f} m", f"تغییر نسبت به شروع: {diff_6m_cm:+.1f} cm", "#dcfce7" if diff_6m_cm >= 0 else "#fee2e2", "#166534" if diff_6m_cm >= 0 else "#991b1b"),
    ("🎯 پیش‌بینی افق ۱۲ ماهه (P50 - سپتامبر ۲۰۲۷)", f"{m12_row['Scenario_P50_m']:.3f} m", f"تغییر نسبت به شروع: {diff_12m_cm:+.1f} cm", "#dcfce7" if diff_12m_cm >= 0 else "#fee2e2", "#166534" if diff_12m_cm >= 0 else "#991b1b"),
    ("💧 مجموع ورودی رودخانه‌ها (۱۲ ماه آینده)", f"{tot_12m_inflow_km3:.1f} km³", f"سناریوی دبی ورودی: {inflow_pct_change:+d}%", "#dbeafe", "#1e40af"),
    (f"⚠️ بیشینه ریسک عبور از آستانه بحران ({crit_thresh:.2f}m)", f"{max_crit_risk_pct:.1f}%", "در ماه دوازدهم افق پیش‌بینی", "#fef3c7" if max_crit_risk_pct < 25 else "#fee2e2", "#92400e" if max_crit_risk_pct < 25 else "#991b1b"),
]

for col_obj, (title_txt, val_txt, sub_txt, bg_c, fg_c) in zip(kpi_cols, cards_data):
    with col_obj:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">{title_txt}</div>
                <div class="kpi-value">{val_txt}</div>
                <div class="kpi-delta" style="background-color:{bg_c}; color:{fg_c};">{sub_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# ۶. تب‌های ۵گانه برجسته (دکمه‌ای با فونت درشت)
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 ۱. پیش‌بینی ۱۲ ماهه تراز خزر",
    "🏞️ ۲. پیش‌بینی دبی رودخانه‌ها (GloFAS + LSTM)",
    "⚖️ ۳. بیلان آب و دورپیوندهای اقلیمی",
    "🎯 ۴. ارزیابی پس‌نگر و کارایی (Skill)",
    "📥 ۵. دانلود گزارش فصلی و خروجی‌های ملی"
])

# ------------------------------------------------------------------------------
# تب ۱: پیش‌بینی احتمالاتی ۱۲ ماهه تراز آب دریای خزر
# ------------------------------------------------------------------------------
with tab1:
    st.markdown(
        """
        <div class="rtl-box">
        <b>تحلیل مدیریتی تراز ۱۲ ماهه:</b> نمودار بادبزنی (Fan Chart) زیر، پیش‌بینی احتمالاتی تراز سطح آب دریای خزر را از <b>اکتبر ۲۰۲۶ تا سپتامبر ۲۰۲۷</b> به همراه باندهای عدم‌قطعیت <b>P10 تا P90</b> (بازه اطمینان ۸۰٪) و <b>P25 تا P75</b> (دامنه میان‌چارکی) نشان می‌دهد. در صورت تغییر اسلایدرهای سناریوی مدیریتی در منوی سمت راست/چپ، منحنی خط‌چین قرمز رفتار تراز تحت سناریوی جدید را نمایش می‌دهد.
        </div>
        """,
        unsafe_allow_html=True
    )

    df_hist_sub = df_master[df_master["Year"] >= hist_start_year].copy()
    f_dates = pd.concat([pd.Series([df_hist_sub.iloc[-1]["Date"]]), df_csl_12m["Date"]]).reset_index(drop=True)
    p10_arr = np.concatenate([[initial_csl_m], df_csl_12m["CSL_P10_m (Low_Level_Risk)"].values])
    p25_arr = np.concatenate([[initial_csl_m], df_csl_12m["CSL_P25_m"].values])
    p50_arr = np.concatenate([[initial_csl_m], df_csl_12m["CSL_P50_m (Median_Forecast)"].values])
    p75_arr = np.concatenate([[initial_csl_m], df_csl_12m["CSL_P75_m"].values])
    p90_arr = np.concatenate([[initial_csl_m], df_csl_12m["CSL_P90_m (High_Level)"].values])
    scen_arr = np.concatenate([[initial_csl_m], df_csl_12m["Scenario_P50_m"].values])

    if PLOTLY_AVAILABLE:
        fig_fan = go.Figure()
        fig_fan.add_trace(go.Scatter(
            x=df_hist_sub["Date"], y=df_hist_sub["CSL_Satellite_Altimetry_m"],
            mode="lines", name="تراز تاریخی / آلتیمتری ماهواره‌ای",
            line=dict(color="#0f172a", width=2.6)
        ))
        fig_fan.add_trace(go.Scatter(
            x=pd.concat([f_dates, f_dates[::-1]]),
            y=np.concatenate([p90_arr, p10_arr[::-1]]),
            fill="toself", fillcolor="rgba(59, 130, 246, 0.22)",
            line=dict(color="rgba(255,255,255,0)"),
            name="باند عدم‌قطعیت P10 تا P90"
        ))
        fig_fan.add_trace(go.Scatter(
            x=pd.concat([f_dates, f_dates[::-1]]),
            y=np.concatenate([p75_arr, p25_arr[::-1]]),
            fill="toself", fillcolor="rgba(37, 99, 235, 0.42)",
            line=dict(color="rgba(255,255,255,0)"),
            name="باند میان‌چارکی P25 تا P75"
        ))
        fig_fan.add_trace(go.Scatter(
            x=f_dates, y=p50_arr,
            mode="lines+markers", name="پیش‌بینی میانه پایه (P50)",
            line=dict(color="#1e3a8a", width=3.2)
        ))
        if inflow_pct_change != 0 or evap_pct_change != 0 or karabogaz_open_pct != 100:
            fig_fan.add_trace(go.Scatter(
                x=f_dates, y=scen_arr,
                mode="lines+markers", name="پیش‌بینی میانه تحت سناریوی انتخابی (What-If)",
                line=dict(color="#dc2626", width=3, dash="dash")
            ))
        fig_fan.add_hline(y=warn_thresh, line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"آستانه هشدار ({warn_thresh:.2f} m)", annotation_position="bottom left")
        fig_fan.add_hline(y=crit_thresh, line_dash="dot", line_color="#dc2626",
                          annotation_text=f"آستانه بحرانی ({crit_thresh:.2f} m)", annotation_position="bottom left")
        fig_fan.update_layout(
            font=dict(family="Vazirmatn, Tahoma"),
            title="نمودار بادبزنی (Fan Chart) پیش‌بینی احتمالاتی ۱۲ ماهه تراز آب دریای خزر (مبنای بالتیک)",
            xaxis_title="تاریخ (میلادی)",
            yaxis_title="تراز سطح آب (m BS)",
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fan, use_container_width=True)

    col_t1_left, col_t1_right = st.columns([1.35, 1.0])
    with col_t1_left:
        st.markdown("##### 📋 جدول کامل پیش‌بینی احتمالاتی ۱۲ ماهه تراز و ریسک عبور از آستانه‌ها")
        display_csl_cols = [
            "Date", "Lead_Month", "CSL_P10_m (Low_Level_Risk)", "CSL_P50_m (Median_Forecast)",
            "CSL_P90_m (High_Level)", "Scenario_P50_m", "Uncertainty_Width_P90_P10_cm",
            "Dynamic_Risk_Warn (%)", "Dynamic_Risk_Crit (%)"
        ]
        st.dataframe(
            df_csl_12m[display_csl_cols].assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d")),
            use_container_width=True, height=410
        )

    with col_t1_right:
        st.markdown("##### 📊 توزیع احتمالاتی ۱۰۰۰ مسیر مونت‌کارلو در افق ماه انتخابی")
        selected_lead = st.select_slider(
            "انتخاب ماه افق پیش‌بینی (ماه ۱ تا ۱۲):",
            options=list(range(1, 13)), value=12
        )
        mc_slice = csl_mc_adj[:, selected_lead - 1]
        target_dt_str = df_csl_12m.loc[selected_lead - 1, "Date"].strftime("%Y-%m")

        if PLOTLY_AVAILABLE:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=mc_slice, nbinsx=35, name="توزیع ۱۰۰۰ عضو مونت‌کارلو",
                marker_color="#3b82f6", opacity=0.78
            ))
            fig_hist.add_vline(x=float(np.percentile(mc_slice, 10)), line_dash="dot", line_color="#1d4ed8", annotation_text="P10")
            fig_hist.add_vline(x=float(np.percentile(mc_slice, 50)), line_dash="solid", line_color="#0f172a", annotation_text="P50")
            fig_hist.add_vline(x=float(np.percentile(mc_slice, 90)), line_dash="dot", line_color="#1d4ed8", annotation_text="P90")
            fig_hist.add_vline(x=crit_thresh, line_dash="dash", line_color="#dc2626", annotation_text="آستانه بحران")
            fig_hist.update_layout(
                font=dict(family="Vazirmatn, Tahoma"),
                title=f"توزیع احتمالاتی تراز در ماه {selected_lead}ام ({target_dt_str})",
                xaxis_title="تراز سطح آب (m BS)", yaxis_title="فراوانی (از ۱۰۰۰ شبیه‌سازی)",
                height=340
            )
            st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------------------------------------------------------
# تب ۲: پیش‌بینی دبی رودخانه‌های ورودی (GloFAS + LSTM) و نقشه مصب‌ها
# ------------------------------------------------------------------------------
with tab2:
    st.markdown(
        """
        <div class="rtl-box">
        <b>معماری ترکیبی پیش‌بینی دبی (Ensemble):</b> در افق کوتاه‌مدت تا میان‌مدت (ماه‌های ۱ تا ۶)، سامانه بر ترکیب خروجی‌های احتمالاتی <b>GloFAS Seasonal (مدل LISFLOOD)</b> و شبکه عصبی <b>LSTM</b> تکیه دارد و در ماه‌های ۷ تا ۱۲، وزن اصلی به مدل یادگیری عمیق آموزش‌دیده با شاخص‌های دورپیوندی و اقلیم‌شناسی حوضه منتقل می‌شود.
        </div>
        """,
        unsafe_allow_html=True
    )

    col_t2_map, col_t2_chart = st.columns([1.0, 1.45])
    with col_t2_map:
        st.markdown("##### 🗺️ موقعیت ۱۰ ایستگاه مرجع مصب رودخانه‌های دریای خزر")
        df_map_pts = df_outlets.rename(columns={"GloFAS_Pixel_Lat": "lat", "GloFAS_Pixel_Lon": "lon"})
        st.map(df_map_pts, latitude="lat", longitude="lon", size="Mean_Discharge_m3s", zoom=4)
        st.dataframe(
            df_outlets[["Station_ID", "River_Name", "Basin_Sector", "GloFAS_Pixel_Lat", "GloFAS_Pixel_Lon", "Mean_Discharge_m3s"]],
            use_container_width=True, height=230
        )

    with col_t2_chart:
        st.markdown("##### 📈 پیش‌بینی ۱۲ ماهه کل دبی ورودی به خزر (مقایسه GloFAS، LSTM و مدل ترکیبی)")
        df_glo_tot = df_seas_6m[df_seas_6m["Station_ID"] != "Volga_Astrakhan"].groupby("Date")[["Q_P10", "Q_P50", "Q_P90"]].sum().reset_index()

        if PLOTLY_AVAILABLE:
            fig_q = go.Figure()
            fig_q.add_trace(go.Scatter(
                x=pd.concat([df_ens_12m["Date"], df_ens_12m["Date"][::-1]]),
                y=np.concatenate([df_ens_12m["Ens_TotalQ_P90_m3s"], df_ens_12m["Ens_TotalQ_P10_m3s"][::-1]]),
                fill="toself", fillcolor="rgba(16, 185, 129, 0.22)",
                line=dict(color="rgba(255,255,255,0)"), name="باند عدم‌قطعیت مدل ترکیبی (P10-P90)"
            ))
            fig_q.add_trace(go.Scatter(
                x=df_ens_12m["Date"], y=df_ens_12m["Ens_TotalQ_P50_m3s"],
                mode="lines+markers", name="مدل ترکیبی نهایی (Hybrid Ensemble P50)",
                line=dict(color="#047857", width=3)
            ))
            fig_q.add_trace(go.Scatter(
                x=df_dl_12m["Date"], y=df_dl_12m["DL_TotalQ_P50"],
                mode="lines+markers", name="شبکه عصبی عمیق (Probabilistic LSTM P50)",
                line=dict(color="#7c3aed", width=2.2, dash="dash")
            ))
            fig_q.add_trace(go.Scatter(
                x=df_glo_tot["Date"], y=df_glo_tot["Q_P50"],
                mode="lines+markers", name="پیش‌بینی فصلی GloFAS P50 (افق ۱ تا ۶ ماه)",
                line=dict(color="#ea580c", width=2.5, dash="dot")
            ))
            fig_q.update_layout(
                font=dict(family="Vazirmatn, Tahoma"),
                xaxis_title="ماه پیش‌بینی", yaxis_title="دبی ورودی کل (m³/s)",
                height=360, hovermode="x unified"
            )
            st.plotly_chart(fig_q, use_container_width=True)

        st.markdown("##### ⚖️ نمودار وزن‌دهی پویا (Skill-Based Weights) در افق ۱ تا ۱۲ ماه")
        if PLOTLY_AVAILABLE:
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(x=df_ens_12m["Lead_Month"], y=df_ens_12m["Weight_GloFAS"], name="وزن سامانه GloFAS", marker_color="#ea580c"))
            fig_w.add_trace(go.Bar(x=df_ens_12m["Lead_Month"], y=df_ens_12m["Weight_DL_LSTM"], name="وزن مدل یادگیری عمیق (LSTM)", marker_color="#7c3aed"))
            fig_w.add_trace(go.Bar(x=df_ens_12m["Lead_Month"], y=df_ens_12m["Weight_Climatology"], name="وزن اقلیم‌شناسی پایه", marker_color="#64748b"))
            fig_w.update_layout(
                font=dict(family="Vazirmatn, Tahoma"),
                barmode="stack", xaxis_title="افق پیش‌بینی (ماه ۱ تا ۱۲)", yaxis_title="سهم وزن در ترکیب (۰ تا ۱)", height=260
            )
            st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🔍 بررسی اختصاصی پیش‌بینی فصلی ۶ ماهه GloFAS به تفکیک هر ایستگاه مصب")
    sel_station = st.selectbox("انتخاب ایستگاه خروجی رودخانه:", options=df_outlets["Station_ID"].tolist(), index=0)
    df_st_6m = df_seas_6m[df_seas_6m["Station_ID"] == sel_station].copy()
    c_st1, c_st2 = st.columns([1.3, 1.0])
    with c_st1:
        if PLOTLY_AVAILABLE:
            fig_st = go.Figure()
            fig_st.add_trace(go.Scatter(
                x=pd.concat([df_st_6m["Date"], df_st_6m["Date"][::-1]]),
                y=np.concatenate([df_st_6m["Q_P90"], df_st_6m["Q_P10"][::-1]]),
                fill="toself", fillcolor="rgba(249, 115, 22, 0.25)", line=dict(color="rgba(0,0,0,0)"),
                name="باند P10 - P90"
            ))
            fig_st.add_trace(go.Scatter(x=df_st_6m["Date"], y=df_st_6m["Q_P50"], mode="lines+markers", name="میانه (P50)", line=dict(color="#c2410c", width=2.5)))
            fig_st.add_trace(go.Scatter(x=df_st_6m["Date"], y=df_st_6m["Q_Mean"], mode="lines", name="میانگین آنسامبل", line=dict(color="#0f172a", dash="dash")))
            fig_st.update_layout(
                font=dict(family="Vazirmatn, Tahoma"),
                title=f"هیدروگراف احتمالاتی ۶ ماهه GloFAS برای ایستگاه {sel_station}", yaxis_title="دبی (m³/s)", height=320
            )
            st.plotly_chart(fig_st, use_container_width=True)
    with c_st2:
        st.dataframe(df_st_6m.assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d")), use_container_width=True, height=280)

# ------------------------------------------------------------------------------
# تب ۳: بیلان آب، منحنی هیپسومتریک و دورپیوندهای اقلیمی (ENSO, NAO, AO)
# ------------------------------------------------------------------------------
with tab3:
    col_b1, col_b2 = st.columns([1.3, 1.0])
    with col_b1:
        st.markdown("##### ⚖️ تفکیک مؤلفه‌های حجمی بیلان آب دریای خزر در ۱۲ ماه آینده (km³/month)")
        if PLOTLY_AVAILABLE:
            fig_wb = go.Figure()
            fig_wb.add_trace(go.Bar(
                x=df_csl_12m["Date"], y=df_csl_12m["Inflow_Vol_km3"],
                name="ورودی رودخانه‌ها (+km³)", marker_color="#2563eb"
            ))
            fig_wb.add_trace(go.Bar(
                x=df_csl_12m["Date"], y=df_csl_12m["Net_PE_Vol_km3"] - df_csl_12m["KaraBogaz_Outflow_km3"],
                name="خالص بارش منهای تبخیر و خروجی قره‌بوغاز (-km³)", marker_color="#e11d48"
            ))
            fig_wb.add_trace(go.Scatter(
                x=df_csl_12m["Date"], y=df_csl_12m["Net_Storage_Change_km3"],
                mode="lines+markers", name="تغییر خالص حجم مخزن (ΔV km³)",
                line=dict(color="#0f172a", width=3)
            ))
            fig_wb.update_layout(
                font=dict(family="Vazirmatn, Tahoma"),
                barmode="group", yaxis_title="حجم آب (کیلومترمکعب در ماه)", height=380, hovermode="x unified"
            )
            st.plotly_chart(fig_wb, use_container_width=True)

    with col_b2:
        st.markdown("##### 📐 جدول منحنی هیپسومتریک دریای خزر (تراز – مساحت – حجم – قره‌بوغاز)")
        st.dataframe(df_hypso, use_container_width=True, height=340)

    st.markdown("---")
    st.markdown("##### 🌍 تحلیل همبستگی تاخیری شاخص‌های اقلیمی دورپیوندی (ENSO، NAO و AO) با دبی ولگا")
    col_tc1, col_tc2 = st.columns([1.2, 1.0])
    with col_tc1:
        df_tele_recent = df_master[df_master["Year"] >= 2015]
        if PLOTLY_AVAILABLE:
            fig_tc = go.Figure()
            fig_tc.add_trace(go.Scatter(x=df_tele_recent["Date"], y=df_tele_recent["ENSO_Nino34"], name="ENSO (Niño 3.4)", line=dict(color="#dc2626", width=2)))
            fig_tc.add_trace(go.Scatter(x=df_tele_recent["Date"], y=df_tele_recent["NAO"], name="NAO Index", line=dict(color="#059669", width=1.6)))
            fig_tc.add_trace(go.Scatter(x=df_tele_recent["Date"], y=df_tele_recent["AO"], name="AO Index", line=dict(color="#4f46e5", width=1.6)))
            fig_tc.update_layout(
                font=dict(family="Vazirmatn, Tahoma"),
                title="سری زمانی شاخص‌های دورپیوندی (۲۰۱۵ تا ۲۰۲۶)", yaxis_title="مقدار شاخص استاندارد", height=340
            )
            st.plotly_chart(fig_tc, use_container_width=True)
    with col_tc2:
        if PLOTLY_AVAILABLE:
            fig_lag = px.bar(
                df_lag_corr, x="Lag_Months", y="Corr_with_Volga_Q", color="Teleconnection",
                barmode="group", title="ضریب همبستگی تاخیری (تاخیر ۱ تا ۱۲ ماه) با دبی رودخانه ولگا"
            )
            fig_lag.update_layout(font=dict(family="Vazirmatn, Tahoma"), height=340)
            st.plotly_chart(fig_lag, use_container_width=True)

# ------------------------------------------------------------------------------
# تب ۴: ارزیابی عملکرد پس‌نگر (Hindcast Skill Evaluation) در افق ۱ تا ۱۲ ماه
# ------------------------------------------------------------------------------
with tab4:
    st.markdown(
        """
        <div class="rtl-box">
        <b>ارزیابی کمی کارایی سامانه (Hindcast Skill):</b> در این بخش، خطای پیش‌بینی سامانه ترکیبی (GloFAS + LSTM) در افق‌های ۱ تا ۱۲ ماهه با دو سناریوی مرجع <b>پایداری (Persistence)</b> و <b>اقلیم‌شناسی (Climatology)</b> مقایسه شده است.
        </div>
        """,
        unsafe_allow_html=True
    )

    col_sk1, col_sk2 = st.columns(2)
    with col_sk1:
        if PLOTLY_AVAILABLE:
            fig_rmse = go.Figure()
            fig_rmse.add_trace(go.Scatter(x=df_skill["Lead_Month"], y=df_skill["CSL_RMSE_Hybrid_cm"], mode="lines+markers", name="سامانه ترکیبی (GloFAS + LSTM)", line=dict(color="#1e3a8a", width=3)))
            fig_rmse.add_trace(go.Scatter(x=df_skill["Lead_Month"], y=df_skill["CSL_RMSE_Climatology_cm"], mode="lines+markers", name="مدل مرجع اقلیم‌شناسی (Climatology)", line=dict(color="#059669", dash="dash", width=2.2)))
            fig_rmse.add_trace(go.Scatter(x=df_skill["Lead_Month"], y=df_skill["CSL_RMSE_Persistence_cm"], mode="lines+markers", name="مدل مرجع پایداری (Persistence)", line=dict(color="#dc2626", dash="dot", width=2.2)))
            fig_rmse.update_layout(
                font=dict(family="Vazirmatn, Tahoma"),
                title="مقایسه خطای RMSE پیش‌بینی تراز دریای خزر (سانتی‌متر) در افق‌های ۱ تا ۱۲ ماه", xaxis_title="افق پیش‌بینی (ماه)", yaxis_title="RMSE (cm)", height=380
            )
            st.plotly_chart(fig_rmse, use_container_width=True)

    with col_sk2:
        if PLOTLY_AVAILABLE:
            fig_ss = go.Figure()
            fig_ss.add_trace(go.Bar(x=df_skill["Lead_Month"], y=df_skill["CSL_Skill_vs_Persistence (%)"], name="بهبود Skill نسبت به پایداری (%)", marker_color="#2563eb"))
            fig_ss.add_trace(go.Bar(x=df_skill["Lead_Month"], y=df_skill["CSL_Skill_vs_Climatology (%)"], name="بهبود Skill نسبت به اقلیم‌شناسی (%)", marker_color="#10b981"))
            fig_ss.update_layout(
                font=dict(family="Vazirmatn, Tahoma"),
                barmode="group", title="درصد ارتقای کارایی (Skill Score %) سامانه نسبت به مدل‌های پایه", xaxis_title="افق پیش‌بینی (ماه)", yaxis_title="Skill Score (%)", height=380
            )
            st.plotly_chart(fig_ss, use_container_width=True)

    st.markdown("##### 📊 جدول کامل شاخص‌های ارزیابی پس‌نگر (RMSE، MAE، NSE و Skill Score) در افق‌های ۱ تا ۱۲ ماهه")
    st.dataframe(df_skill, use_container_width=True, height=420)

# ------------------------------------------------------------------------------
# تب ۵: درگاه دانلود گزارش فصلی خودکار و خروجی‌های یکپارچه‌سازی ملی (API)
# ------------------------------------------------------------------------------
with tab5:
    st.markdown(
        """
        <div class="rtl-box">
        <b>یکپارچه‌سازی با سامانه‌های ملی و تولید گزارش‌های دوره‌ای خودکار:</b> مطابق بخش ۴-۴ پروپوزال، تمامی خروجی‌های سامانه در قالب فرمت‌های استاندارد تبادل داده (<b>JSON</b>، <b>GeoJSON</b> و <b>CSV</b>) و همچنین <b>گزارش مدیریتی فصلی خودکار (قابل چاپ به PDF)</b> از طریق دکمه‌های زیر قابل دریافت مستقیم هستند.
        </div>
        """,
        unsafe_allow_html=True
    )

    api_payload = {
        "system": "Caspian Sea 12-Month Probabilistic Forecasting System (Phase 2)",
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "initial_sea_level_m_bs": initial_csl_m,
        "scenario_parameters": {
            "inflow_change_pct": inflow_pct_change,
            "evaporation_change_pct": evap_pct_change,
            "karabogaz_open_pct": karabogaz_open_pct,
            "warning_threshold_m": warn_thresh,
            "critical_threshold_m": crit_thresh,
        },
        "forecasts_12m": df_csl_12m.assign(Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
        "hindcast_skill": df_skill.to_dict(orient="records"),
    }
    json_bytes = json.dumps(api_payload, indent=2, ensure_ascii=False).encode("utf-8")

    geojson_obj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(r["GloFAS_Pixel_Lon"]), float(r["GloFAS_Pixel_Lat"])]},
                "properties": r.to_dict()
            }
            for _, r in df_outlets.iterrows()
        ]
    }
    geojson_bytes = json.dumps(geojson_obj, indent=2, ensure_ascii=False).encode("utf-8")

    html_report = f"""
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>گزارش فصلی پیش‌بینی ۱۲ ماهه تراز آب دریای خزر</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap');
            body {{ font-family: 'Vazirmatn', Tahoma, sans-serif; margin: 35px; color: #1e293b; line-height: 1.8; }}
            h1, h2 {{ color: #0f172a; border-bottom: 2px solid #1d4ed8; padding-bottom: 6px; }}
            .summary-box {{ background: #f1f5f9; padding: 16px 20px; border-right: 5px solid #1d4ed8; border-radius: 8px; margin-bottom: 22px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
            th, td {{ border: 1px solid #cbd5e1; padding: 9px; text-align: center; }}
            th {{ background-color: #1e3a8a; color: white; }}
        </style>
    </head>
    <body>
        <h1>گزارش فصلی پایش و پیش‌بینی احتمالاتی ۱۲ ماهه تراز آب دریای خزر</h1>
        <p><b>مرجع صادرکننده:</b> سامانه هوشمند پیش‌بینی تراز دریای خزر (فاز دوم) - مرکز مطالعات و تحقیقات دریای خزر</p>
        <div class="summary-box">
            <h3>خلاصه مدیریتی (Executive Summary)</h3>
            <ul>
                <li><b>تراز اولیه دریا (سپتامبر ۲۰۲۶):</b> {initial_csl_m:.3f} متر (مبنای بالتیک)</li>
                <li><b>پیش‌بینی میانه افق ۶ ماهه (مارس ۲۰۲۷):</b> {m6_row['Scenario_P50_m']:.3f} متر (بازه P10 تا P90: {m6_row['Scenario_P10_m']:.3f} تا {m6_row['Scenario_P90_m']:.3f} متر)</li>
                <li><b>پیش‌بینی میانه افق ۱۲ ماهه (سپتامبر ۲۰۲۷):</b> {m12_row['Scenario_P50_m']:.3f} متر (بازه P10 تا P90: {m12_row['Scenario_P10_m']:.3f} تا {m12_row['Scenario_P90_m']:.3f} متر)</li>
                <li><b>مجموع حجم رواناب ورودی پیش‌بینی‌شده در ۱۲ ماه آینده:</b> {tot_12m_inflow_km3:.2f} کیلومترمکعب</li>
                <li><b>بیشینه احتمال عبور از آستانه بحرانی ({crit_thresh:.2f} متر):</b> {max_crit_risk_pct:.1f}%</li>
            </ul>
        </div>
        <h2>جدول پیش‌بینی احتمالاتی ۱۲ ماهه تراز و مؤلفه‌های بیلان آب</h2>
        {df_csl_12m[['Date', 'Lead_Month', 'Inflow_Vol_km3', 'Net_PE_Vol_km3', 'CSL_P10_m (Low_Level_Risk)', 'CSL_P50_m (Median_Forecast)', 'CSL_P90_m (High_Level)', 'Scenario_P50_m', 'Dynamic_Risk_Warn (%)', 'Dynamic_Risk_Crit (%)']].assign(Date=lambda d: d['Date'].dt.strftime('%Y-%m-%d')).to_html(index=False)}
    </body>
    </html>
    """

    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
    with dl_col1:
        st.download_button(
            label="📄 ۱. دانلود گزارش فصلی خودکار (HTML/PDF)",
            data=html_report.encode("utf-8"),
            file_name="Caspian_Seasonal_Executive_Report.html",
            mime="text/html",
            use_container_width=True
        )
    with dl_col2:
        st.download_button(
            label="📊 ۲. دانلود جدول پیش‌بینی ۱۲ ماهه (CSV)",
            data=df_csl_12m.to_csv(index=False).encode("utf-8"),
            file_name="Caspian_12M_Probabilistic_Forecast.csv",
            mime="text/csv",
            use_container_width=True
        )
    with dl_col3:
        st.download_button(
            label="🗺️ ۳. دانلود لایه مکانی مصب‌ها (GeoJSON)",
            data=geojson_bytes,
            file_name="caspian_river_outlets.geojson",
            mime="application/geo+json",
            use_container_width=True
        )
    with dl_col4:
        st.download_button(
            label="📥 ۴. دانلود خروجی وب‌سرویس (JSON API)",
            data=json_bytes,
            file_name="caspian_12m_forecast_api.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("##### 📌 خلاصه مدیریتی فصلی (۴ فصل پیش‌رو بر اساس خروجی مدل ۱۲ ماهه)")
    df_seasonal_summary = pd.DataFrame([
        {
            "فصل پیش‌بینی": "پاییز (اکتبر تا دسامبر ۲۰۲۶)",
            "افق زمانی": "ماه‌های ۱ تا ۳",
            "حجم ورودی رودخانه‌ها (km³)": round(float(df_csl_12m.iloc[0:3]["Inflow_Vol_km3"].sum()), 2),
            "خالص بارش منهای تبخیر (km³)": round(float(df_csl_12m.iloc[0:3]["Net_PE_Vol_km3"].sum()), 2),
            "تراز میانه پایان فصل P50 (m BS)": float(df_csl_12m.iloc[2]["Scenario_P50_m"]),
            "بازه عدم‌قطعیت P10 تا P90 (m BS)": f"{df_csl_12m.iloc[2]['Scenario_P10_m']:.3f} تا {df_csl_12m.iloc[2]['Scenario_P90_m']:.3f}",
        },
        {
            "فصل پیش‌بینی": "زمستان (ژانویه تا مارس ۲۰۲۷)",
            "افق زمانی": "ماه‌های ۴ تا ۶",
            "حجم ورودی رودخانه‌ها (km³)": round(float(df_csl_12m.iloc[3:6]["Inflow_Vol_km3"].sum()), 2),
            "خالص بارش منهای تبخیر (km³)": round(float(df_csl_12m.iloc[3:6]["Net_PE_Vol_km3"].sum()), 2),
            "تراز میانه پایان فصل P50 (m BS)": float(df_csl_12m.iloc[5]["Scenario_P50_m"]),
            "بازه عدم‌قطعیت P10 تا P90 (m BS)": f"{df_csl_12m.iloc[5]['Scenario_P10_m']:.3f} تا {df_csl_12m.iloc[5]['Scenario_P90_m']:.3f}",
        },
        {
            "فصل پیش‌بینی": "بهار (آوریل تا ژوئن ۲۰۲۷ - اوج سیلاب ولگا)",
            "افق زمانی": "ماه‌های ۷ تا ۹",
            "حجم ورودی رودخانه‌ها (km³)": round(float(df_csl_12m.iloc[6:9]["Inflow_Vol_km3"].sum()), 2),
            "خالص بارش منهای تبخیر (km³)": round(float(df_csl_12m.iloc[6:9]["Net_PE_Vol_km3"].sum()), 2),
            "تراز میانه پایان فصل P50 (m BS)": float(df_csl_12m.iloc[8]["Scenario_P50_m"]),
            "بازه عدم‌قطعیت P10 تا P90 (m BS)": f"{df_csl_12m.iloc[8]['Scenario_P10_m']:.3f} تا {df_csl_12m.iloc[8]['Scenario_P90_m']:.3f}",
        },
        {
            "فصل پیش‌بینی": "تابستان (ژوئیه تا سپتامبر ۲۰۲۷ - اوج تبخیر)",
            "افق زمانی": "ماه‌های ۱۰ تا ۱۲",
            "حجم ورودی رودخانه‌ها (km³)": round(float(df_csl_12m.iloc[9:12]["Inflow_Vol_km3"].sum()), 2),
            "خالص بارش منهای تبخیر (km³)": round(float(df_csl_12m.iloc[9:12]["Net_PE_Vol_km3"].sum()), 2),
            "تراز میانه پایان فصل P50 (m BS)": float(df_csl_12m.iloc[11]["Scenario_P50_m"]),
            "بازه عدم‌قطعیت P10 تا P90 (m BS)": f"{df_csl_12m.iloc[11]['Scenario_P10_m']:.3f} تا {df_csl_12m.iloc[11]['Scenario_P90_m']:.3f}",
        },
    ])
    st.dataframe(df_seasonal_summary, use_container_width=True)

    with st.expander("💻 مشاهده ساختار فنی بسته وب‌سرویس (REST API JSON Payload ویژه اتصال به سامانه‌های ملی)"):
        st.json(api_payload, expanded=False)
