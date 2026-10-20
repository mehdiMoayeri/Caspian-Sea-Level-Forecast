import os
import json
import pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# بررسی وجود کتابخانه Plotly برای نمودارهای تعاملی؛ در صورت عدم نصب، استفاده خودکار از Matplotlib
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ==============================================================================
# ۱. تنظیمات صفحه و استایل راست‌به‌چپ (RTL) فارسی برای داشبورد مدیریتی
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
    .main-header {
        background: linear-gradient(90deg, #0a2540 0%, #1d4ed8 100%);
        color: white;
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        direction: rtl;
        text-align: right;
    }
    .main-header h1 {
        color: white !important;
        font-size: 1.65rem;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #e2e8f0;
        font-size: 0.95rem;
        margin: 0;
    }
    .rtl-box {
        direction: rtl;
        text-align: right;
        background-color: #f8fafc;
        border-right: 5px solid #1d4ed8;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
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
        <p>مرکز مطالعات و تحقیقات دریای خزر | معماری ترکیبی: سامانه دینامیکی GloFAS v4.0 (افق ۱ تا ۶ ماه) + شبکه عصبی عمیق LSTM (افق ۱ تا ۱۲ ماه) + مدل بیلان آب و منحنی هیپسومتریک</p>
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
    # اطمینان از فرمت تاریخ در تمام جداول
    for k in ["historical_master", "glofas_seasonal_6m", "dl_forecast_12m", "ensemble_inflow_12m", "csl_forecast_12m"]:
        if k in data and "Date" in data[k].columns:
            data[k]["Date"] = pd.to_datetime(data[k]["Date"])
    return data

default_paths = [
    Path("./Caspian_Phase2_Database/08_API_JSON_GeoJSON_NetCDF/caspian_streamlit_bundle.pkl"),
    Path("./caspian_streamlit_bundle.pkl"),
    Path("../Caspian_Phase2_Database/08_API_JSON_GeoJSON_NetCDF/caspian_streamlit_bundle.pkl"),
]

bundle = None
loaded_path_str = None
for p in default_paths:
    if p.is_file():
        bundle = load_bundle_from_path(str(p))
        loaded_path_str = str(p.resolve())
        break

with st.sidebar:
    st.markdown("### ⚙️ تنظیمات سامانه و سناریوها")
    if bundle is None:
        st.warning("فایل `caspian_streamlit_bundle.pkl` در مسیر پیش‌فرض یافت نشد. لطفاً آن را در کادر زیر آپلود کنید:")
        uploaded_pkl = st.file_uploader("انتخاب فایل caspian_streamlit_bundle.pkl", type=["pkl"])
        if uploaded_pkl is not None:
            bundle = pickle.load(uploaded_pkl)
            for k in ["historical_master", "glofas_seasonal_6m", "dl_forecast_12m", "ensemble_inflow_12m", "csl_forecast_12m"]:
                if k in bundle and "Date" in bundle[k].columns:
                    bundle[k]["Date"] = pd.to_datetime(bundle[k]["Date"])
    else:
        st.success("✅ متصل به پایگاه داده یکپارچه فاز دوم")

if bundle is None:
    st.info("لطفاً فایل `caspian_streamlit_bundle.pkl` را کنار فایل `app.py` قرار دهید یا از منوی سمت چپ بارگذاری نمایید.")
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
csl_mc_base = bundle["csl_mc_paths"]  # ابعاد: (1000, 12)

initial_csl_m = float(df_master.iloc[-1]["CSL_Satellite_Altimetry_m"])

# ==============================================================================
# ۳. کنترلرهای تعاملی نوار کناری (آستانه‌های ریسک و شبیه‌ساز سناریوی What-If)
# ==============================================================================
with st.sidebar:
    st.markdown("---")
    st.markdown("#### 🚨 ۱. تنظیم آستانه‌های هشدار و بحران تراز")
    warn_thresh = st.slider(
        "آستانه هشدار زرد (m BS):",
        min_value=-29.60, max_value=-28.80, value=-29.30, step=0.01, format="%.2f"
    )
    crit_thresh = st.slider(
        "آستانه بحرانی قرمز (m BS):",
        min_value=-29.70, max_value=-28.90, value=-29.38, step=0.01, format="%.2f"
    )

    st.markdown("---")
    st.markdown("#### 🎛️ ۲. شبیه‌ساز سناریوهای مدیریتی (What-If)")
    st.caption("تغییر پارامترهای ورودی برای بررسی حساسیت تراز خزر در ۱۲ ماه آینده:")
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
# ۴. محاسبه زنده سناریوی تعاملی (What-If) و به‌روزرسانی احتمال ریسک‌ها
# ==============================================================================
def get_area_km2(h_m):
    return float(np.interp(h_m, df_hypso["Elevation_m_BS"], df_hypso["Area_1000_km2"]) * 1000.0)

# محاسبه مجدد تغییرات تراز تحت سناریوی انتخابی کاربر در نوار کناری
scenario_delta_h = np.zeros(len(df_csl_12m))
cum_dh_scen = 0.0
for idx, row in df_csl_12m.iterrows():
    h_ref = row["CSL_P50_m (Median_Forecast)"]
    area_k = get_area_km2(h_ref)
    
    # حجم اضافه/کاهش یافته ناشی از تغییر درصد دبی، تبخیر و خروجی قره‌بوغاز
    dq_vol  = row["Inflow_Vol_km3"] * (inflow_pct_change / 100.0)
    # برآورد حجم تبخیر پایه ماهانه از روی اقلیم‌شناسی
    m_num = row["Date"].month
    clim_evap_mm = float(df_clim.loc[df_clim["Month"] == m_num, "Clim_Evap_mm"].values[0]) * 1.56
    evap_vol_base = (clim_evap_mm / 1e6) * area_k
    de_vol  = -evap_vol_base * (evap_pct_change / 100.0)
    dkb_vol = -row["KaraBogaz_Outflow_km3"] * ((karabogaz_open_pct - 100.0) / 100.0)
    
    dV_scen = dq_vol + de_vol + dkb_vol
    cum_dh_scen += (dV_scen / area_k) * 1000.0
    scenario_delta_h[idx] = cum_dh_scen

# اعمال شیفت سناریو روی ۱۰۰۰ مسیر مونت‌کارلو و جدول پیش‌بینی ۱۲ ماهه
csl_mc_adj = csl_mc_base + scenario_delta_h[np.newaxis, :]
df_csl_12m["Scenario_P50_m"] = np.round(df_csl_12m["CSL_P50_m (Median_Forecast)"] + scenario_delta_h, 3)
df_csl_12m["Scenario_P10_m"] = np.round(df_csl_12m["CSL_P10_m (Low_Level_Risk)"] + scenario_delta_h, 3)
df_csl_12m["Scenario_P90_m"] = np.round(df_csl_12m["CSL_P90_m (High_Level)"] + scenario_delta_h, 3)
df_csl_12m["Dynamic_Risk_Warn (%)"] = [round(float(np.mean(csl_mc_adj[:, i] < warn_thresh) * 100.0), 1) for i in range(12)]
df_csl_12m["Dynamic_Risk_Crit (%)"] = [round(float(np.mean(csl_mc_adj[:, i] < crit_thresh) * 100.0), 1) for i in range(12)]

# ==============================================================================
# ۵. کارت‌های شاخص کلیدی عملکرد (Executive KPIs) در بالای داشبورد
# ==============================================================================
m6_row  = df_csl_12m.iloc[5]   # ماه ۶ پیش‌بینی (مارس ۲۰۲۷)
m12_row = df_csl_12m.iloc[11]  # ماه ۱۲ پیش‌بینی (سپتامبر ۲۰۲۷)
tot_12m_inflow_km3 = float(df_csl_12m["Inflow_Vol_km3"].sum()) * (1.0 + inflow_pct_change / 100.0)
max_crit_risk_pct  = float(df_csl_12m["Dynamic_Risk_Crit (%)"].max())

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric(
        label="📍 تراز اولیه دریا (سپتامبر ۲۰۲۶)",
        value=f"{initial_csl_m:.3f} m",
        delta="مبنای بالتیک (BS)",
        delta_color="off"
    )
with kpi2:
    diff_6m_cm = (m6_row["Scenario_P50_m"] - initial_csl_m) * 100.0
    st.metric(
        label="🔮 پیش‌بینی افق ۶ ماهه (P50 - مارس ۲۰۲۷)",
        value=f"{m6_row['Scenario_P50_m']:.3f} m",
        delta=f"{diff_6m_cm:+.1f} cm نسبت به شروع"
    )
with kpi3:
    diff_12m_cm = (m12_row["Scenario_P50_m"] - initial_csl_m) * 100.0
    st.metric(
        label="🎯 پیش‌بینی افق ۱۲ ماهه (P50 - سپتامبر ۲۰۲۷)",
        value=f"{m12_row['Scenario_P50_m']:.3f} m",
        delta=f"{diff_12m_cm:+.1f} cm نسبت به شروع"
    )
with kpi4:
    st.metric(
        label="💧 مجموع ورودی رودخانه‌ها (۱۲ ماه آینده)",
        value=f"{tot_12m_inflow_km3:.1f} km³",
        delta=f"سناریوی دبی: {inflow_pct_change:+d}%"
    )
with kpi5:
    st.metric(
        label=f"⚠️ بیشینه ریسک عبور از آستانه بحران ({crit_thresh:.2f}m)",
        value=f"{max_crit_risk_pct:.1f}%",
        delta="در ماه دوازدهم افق پیش‌بینی",
        delta_color="inverse"
    )

# ==============================================================================
# ۶. ایجاد ۵ تب تخصصی منطبق بر اهداف پروپوزال
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 ۱. پیش‌بینی ۱۲ ماهه تراز خزر (Fan Chart)",
    "🏞️ ۲. پیش‌بینی دبی رودخانه‌ها (GloFAS + LSTM)",
    "⚖️ ۳. بیلان آب، هیپسومتری و دورپیوندها",
    "🎯 ۴. ارزیابی پس‌نگر و کارایی (Skill Score)",
    "🔌 ۵. خروجی‌های API ملی و گزارش خودکار فصلی"
])

# ------------------------------------------------------------------------------
# تب ۱: پیش‌بینی احتمالاتی ۱۲ ماهه تراز آب دریای خزر و تحلیل عدم‌قطعیت
# ------------------------------------------------------------------------------
with tab1:
    st.markdown(
        """
        <div class="rtl-box">
        <b>تحلیل مدیریتی تراز ۱۲ ماهه:</b> نمودار بادبزنی (Fan Chart) زیر، پیش‌بینی احتمالاتی تراز سطح آب دریای خزر را از <b>اکتبر ۲۰۲۶ تا سپتامبر ۲۰۲۷</b> به همراه باندهای عدم‌قطعیت <b>P10 تا P90</b> (۹۰٪ اطمینان) و <b>P25 تا P75</b> (دامنه میان‌چارکی) نشان می‌دهد. در صورتی که از منوی سمت چپ، سناریوی مدیریتی (What-If) را تغییر دهید، منحنی خط‌چین قرمز رفتار تراز تحت سناریوی جدید را در مقایسه با شرایط پایه نمایش می‌دهد.
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
        # سری تاریخی
        fig_fan.add_trace(go.Scatter(
            x=df_hist_sub["Date"], y=df_hist_sub["CSL_Satellite_Altimetry_m"],
            mode="lines", name="تراز تاریخی / آلتیمتری ماهواره‌ای",
            line=dict(color="#0f172a", width=2.5)
        ))
        # باند P10 - P90
        fig_fan.add_trace(go.Scatter(
            x=pd.concat([f_dates, f_dates[::-1]]),
            y=np.concatenate([p90_arr, p10_arr[::-1]]),
            fill="toself", fillcolor="rgba(59, 130, 246, 0.22)",
            line=dict(color="rgba(255,255,255,0)"),
            name="باند عدم‌قطعیت P10 - P90"
        ))
        # باند P25 - P75
        fig_fan.add_trace(go.Scatter(
            x=pd.concat([f_dates, f_dates[::-1]]),
            y=np.concatenate([p75_arr, p25_arr[::-1]]),
            fill="toself", fillcolor="rgba(37, 99, 235, 0.40)",
            line=dict(color="rgba(255,255,255,0)"),
            name="باند میان‌چارکی P25 - P75"
        ))
        # میانه پایه P50
        fig_fan.add_trace(go.Scatter(
            x=f_dates, y=p50_arr,
            mode="lines+markers", name="پیش‌بینی میانه پایه (P50)",
            line=dict(color="#1e3a8a", width=3)
        ))
        # منحنی سناریوی تعاملی در صورت تغییر اسلایدرها
        if inflow_pct_change != 0 or evap_pct_change != 0 or karabogaz_open_pct != 100:
            fig_fan.add_trace(go.Scatter(
                x=f_dates, y=scen_arr,
                mode="lines+markers", name="پیش‌بینی میانه تحت سناریوی انتخابی (What-If)",
                line=dict(color="#dc2626", width=3, dash="dash")
            ))
        # خطوط آستانه هشدار و بحران
        fig_fan.add_hline(y=warn_thresh, line_dash="dash", line_color="#f59e0b",
                          annotation_text=f"آستانه هشدار ({warn_thresh:.2f} m)", annotation_position="bottom left")
        fig_fan.add_hline(y=crit_thresh, line_dash="dot", line_color="#dc2626",
                          annotation_text=f"آستانه بحرانی ({crit_thresh:.2f} m)", annotation_position="bottom left")
        fig_fan.update_layout(
            title="نمودار بادبزنی (Fan Chart) پیش‌بینی احتمالاتی ۱۲ ماهه تراز آب دریای خزر (مبنای بالتیک)",
            xaxis_title="تاریخ (میلادی)",
            yaxis_title="تراز سطح آب (m BS)",
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_fan, use_container_width=True)
    else:
        fig_m, ax_m = plt.subplots(figsize=(14, 5))
        ax_m.plot(df_hist_sub["Date"], df_hist_sub["CSL_Satellite_Altimetry_m"], color="black", lw=2, label="Observed Sea Level")
        ax_m.fill_between(f_dates, p10_arr, p90_arr, color="dodgerblue", alpha=0.25, label="P10-P90 Band")
        ax_m.fill_between(f_dates, p25_arr, p75_arr, color="royalblue", alpha=0.45, label="P25-P75 Band")
        ax_m.plot(f_dates, p50_arr, color="navy", lw=2.5, marker="o", label="Baseline P50")
        ax_m.plot(f_dates, scen_arr, color="crimson", lw=2.2, ls="--", label="Scenario P50")
        ax_m.axhline(warn_thresh, color="darkorange", ls="--", label=f"Warning ({warn_thresh:.2f}m)")
        ax_m.axhline(crit_thresh, color="red", ls=":", label=f"Critical ({crit_thresh:.2f}m)")
        ax_m.legend()
        ax_m.grid(True, ls="--", alpha=0.5)
        st.pyplot(fig_m)

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
            use_container_width=True, height=420
        )

    with col_t1_right:
        st.markdown("##### 📊 توزیع احتمالاتی ۱۰۰۰ مسیر مونت‌کارلو در افق ماه انتخابی")
        selected_lead = st.select_slider(
            "انتخاب ماه افق پیش‌بینی (Lead Month 1 to 12):",
            options=list(range(1, 13)), value=12
        )
        mc_slice = csl_mc_adj[:, selected_lead - 1]
        target_dt_str = df_csl_12m.loc[selected_lead - 1, "Date"].strftime("%Y-%m")

        if PLOTLY_AVAILABLE:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=mc_slice, nbinsx=35, name="توزیع ۱۰۰۰ عضو مونت‌کارلو",
                marker_color="#3b82f6", opacity=0.75
            ))
            fig_hist.add_vline(x=float(np.percentile(mc_slice, 10)), line_dash="dot", line_color="#1d4ed8", annotation_text="P10")
            fig_hist.add_vline(x=float(np.percentile(mc_slice, 50)), line_dash="solid", line_color="#0f172a", annotation_text="P50")
            fig_hist.add_vline(x=float(np.percentile(mc_slice, 90)), line_dash="dot", line_color="#1d4ed8", annotation_text="P90")
            fig_hist.add_vline(x=crit_thresh, line_dash="dash", line_color="#dc2626", annotation_text="آستانه بحران")
            fig_hist.update_layout(
                title=f"توزیع احتمالاتی تراز در ماه {selected_lead}ام ({target_dt_str})",
                xaxis_title="تراز سطح آب (m BS)", yaxis_title="فراوانی (از ۱۰۰۰ شبیه‌سازی)",
                height=350
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            fig_h, ax_h = plt.subplots(figsize=(6, 4))
            ax_h.hist(mc_slice, bins=30, color="dodgerblue", alpha=0.7, edgecolor="white")
            ax_h.axvline(crit_thresh, color="red", ls="--", label="Critical")
            st.pyplot(fig_h)

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
                xaxis_title="ماه پیش‌بینی", yaxis_title="دبی ورودی کل (m³/s)",
                height=360, hovermode="x unified"
            )
            st.plotly_chart(fig_q, use_container_width=True)

        st.markdown("##### ⚖️ جدول و نمودار وزن‌دهی پویا (Skill-Based Weights) در افق ۱ تا ۱۲ ماه")
        if PLOTLY_AVAILABLE:
            fig_w = go.Figure()
            fig_w.add_trace(go.Bar(x=df_ens_12m["Lead_Month"], y=df_ens_12m["Weight_GloFAS"], name="وزن سامانه GloFAS", marker_color="#ea580c"))
            fig_w.add_trace(go.Bar(x=df_ens_12m["Lead_Month"], y=df_ens_12m["Weight_DL_LSTM"], name="وزن مدل یادگیری عمیق (LSTM)", marker_color="#7c3aed"))
            fig_w.add_trace(go.Bar(x=df_ens_12m["Lead_Month"], y=df_ens_12m["Weight_Climatology"], name="وزن اقلیم‌شناسی پایه", marker_color="#64748b"))
            fig_w.update_layout(barmode="stack", xaxis_title="افق پیش‌بینی (ماه ۱ تا ۱۲)", yaxis_title="سهم وزن در ترکیب (۰ تا ۱)", height=260)
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
            fig_st.update_layout(title=f"هیدروگراف احتمالاتی ۶ ماهه GloFAS برای ایستگاه {sel_station}", yaxis_title="دبی (m³/s)", height=320)
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
            fig_wb.update_layout(barmode="group", yaxis_title="حجم آب (کیلومترمکعب در ماه)", height=380, hovermode="x unified")
            st.plotly_chart(fig_wb, use_container_width=True)

    with col_b2:
        st.markdown("##### 📐 جدول و منحنی هیپسومتریک دریای خزر (تراز – مساحت – حجم – قره‌بوغاز)")
        st.dataframe(df_hypso, use_container_width=True, height=340)

    st.markdown("---")
    st.markdown("##### 🌍 تحلیل همبستگی تاخیری شاخص‌های اقلیمی دورپیوندی (ENSO، NAO و AO) با دبی ولگا و بیلان خزر")
    col_tc1, col_tc2 = st.columns([1.2, 1.0])
    with col_tc1:
        df_tele_recent = df_master[df_master["Year"] >= 2015]
        if PLOTLY_AVAILABLE:
            fig_tc = go.Figure()
            fig_tc.add_trace(go.Scatter(x=df_tele_recent["Date"], y=df_tele_recent["ENSO_Nino34"], name="ENSO (Niño 3.4)", line=dict(color="#dc2626", width=2)))
            fig_tc.add_trace(go.Scatter(x=df_tele_recent["Date"], y=df_tele_recent["NAO"], name="NAO Index", line=dict(color="#059669", width=1.6)))
            fig_tc.add_trace(go.Scatter(x=df_tele_recent["Date"], y=df_tele_recent["AO"], name="AO Index", line=dict(color="#4f46e5", width=1.6)))
            fig_tc.update_layout(title="سری زمانی شاخص‌های دورپیوندی (۲۰۱۵ تا ۲۰۲۶)", yaxis_title="مقدار شاخص استاندارد", height=340)
            st.plotly_chart(fig_tc, use_container_width=True)
    with col_tc2:
        if PLOTLY_AVAILABLE:
            fig_lag = px.bar(
                df_lag_corr, x="Lag_Months", y="Corr_with_Volga_Q", color="Teleconnection",
                barmode="group", title="ضریب همبستگی تاخیری (Lag 1–12 Months) با دبی رودخانه ولگا"
            )
            fig_lag.update_layout(height=340)
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
            fig_rmse.update_layout(title="مقایسه خطای RMSE پیش‌بینی تراز دریای خزر (سانتی‌متر) در افق‌های ۱ تا ۱۲ ماه", xaxis_title="افق پیش‌بینی (ماه)", yaxis_title="RMSE (cm)", height=380)
            st.plotly_chart(fig_rmse, use_container_width=True)

    with col_sk2:
        if PLOTLY_AVAILABLE:
            fig_ss = go.Figure()
            fig_ss.add_trace(go.Bar(x=df_skill["Lead_Month"], y=df_skill["CSL_Skill_vs_Persistence (%)"], name="بهبود Skill نسبت به پایداری (%)", marker_color="#2563eb"))
            fig_ss.add_trace(go.Bar(x=df_skill["Lead_Month"], y=df_skill["CSL_Skill_vs_Climatology (%)"], name="بهبود Skill نسبت به اقلیم‌شناسی (%)", marker_color="#10b981"))
            fig_ss.update_layout(barmode="group", title="درصد ارتقای کارایی (Skill Score %) سامانه نسبت به مدل‌های پایه", xaxis_title="افق پیش‌بینی (ماه)", yaxis_title="Skill Score (%)", height=380)
            st.plotly_chart(fig_ss, use_container_width=True)

    st.markdown("##### 📊 جدول کامل شاخص‌های ارزیابی پس‌نگر (RMSE، MAE، NSE و Skill Score) در افق‌های ۱ تا ۱۲ ماهه")
    st.dataframe(df_skill, use_container_width=True, height=420)

# ------------------------------------------------------------------------------
# تب ۵: درگاه یکپارچه‌سازی ملی (API / JSON / GeoJSON / NetCDF) و گزارش فصلی خودکار
# ------------------------------------------------------------------------------
with tab5:
    st.markdown(
        """
        <div class="rtl-box">
        <b>یکپارچه‌سازی با سامانه‌های ملی و تولید گزارش‌های دوره‌ای خودکار:</b> مطابق بخش ۴-۴ پروپوزال، تمامی خروجی‌های سامانه در قالب فرمت‌های استاندارد وب‌سرویس (<b>REST JSON</b>، <b>GeoJSON</b>، <b>NetCDF</b> و <b>CSV</b>) و همچنین <b>گزارش مدیریتی فصلی خودکار</b> از طریق درگاه زیر قابل دریافت و تبادل مستقیم هستند.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ساخت JSON استاندارد وب‌سرویس به صورت داینامیک
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

    # ساخت GeoJSON مصب رودخانه‌ها
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

    # ساخت گزارش مدیریتی فصلی خودکار (HTML قابل چاپ به PDF در مرورگر)
    html_report = f"""
    <!DOCTYPE html>
    <html lang="fa" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <title>گزارش فصلی پیش‌بینی ۱۲ ماهه تراز آب دریای خزر</title>
        <style>
            body {{ font-family: Tahoma, Arial, sans-serif; margin: 35px; color: #1e293b; line-height: 1.7; }}
            h1, h2 {{ color: #0f172a; border-bottom: 2px solid #1d4ed8; padding-bottom: 6px; }}
            .summary-box {{ background: #f1f5f9; padding: 15px; border-right: 5px solid #1d4ed8; margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
            th, td {{ border: 1px solid #cbd5e1; padding: 8px; text-align: center; }}
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
            label="📥 دانلود خروجی وب‌سرویس (JSON API)",
            data=json_bytes,
            file_name="caspian_12m_forecast_api.json",
            mime="application/json",
            use_container_width=True
        )
    with dl_col2:
        st.download_button(
            label="🗺️ دانلود لایه مکانی مصب‌ها (GeoJSON)",
            data=geojson_bytes,
            file_name="caspian_river_outlets.geojson",
            mime="application/geo+json",
            use_container_width=True
        )
    with dl_col3:
        st.download_button(
            label="📊 دانلود جدول پیش‌بینی ۱۲ ماهه (CSV)",
            data=df_csl_12m.to_csv(index=False).encode("utf-8"),
            file_name="Caspian_12M_Probabilistic_Forecast.csv",
            mime="text/csv",
            use_container_width=True
        )
    with dl_col4:
        st.download_button(
            label="📄 دانلود گزارش فصلی خودکار (HTML/PDF)",
            data=html_report.encode("utf-8"),
            file_name="Caspian_Seasonal_Executive_Report.html",
            mime="text/html",
            use_container_width=True
        )

    st.markdown("##### 💻 پیش‌نمایش زنده خروجی استاندارد REST API (JSON Payload جهت اتصال به سامانه‌های ملی)")
    st.json(api_payload, expanded=False)