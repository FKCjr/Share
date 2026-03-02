import streamlit as st
import pandas as pd
import numpy as np
import pyet
import scipy.stats as stats
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import io
import zipfile

# --- 1. GLOBAL UI CONFIG ---
st.set_page_config(page_title="Luangwa Master Lab 2.0", layout="wide")

# Custom CSS for the "Glossy Blue" Sidebar and Dark Main Area
st.markdown("""
    <style>
    .stApp { background-color: rgba(77, 77, 77, 0.8); color: #ffffff; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(0,77,102,0.6) 0%, rgba(0,30,40,0.9) 100%);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .stTabs [data-baseweb="tab"] { color: #ffffff; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

plt.rcParams.update({
    "font.family": "serif", 
    "font.serif": ["Times New Roman"],
    "text.color": "black",
    "axes.labelcolor": "black"
})

# --- 2. SESSION STATE INIT ---
for key in ['AA', 'AB', 'AC', 'SetE', 'SetF', 'SetG']:
    if key not in st.session_state: st.session_state[key] = {}

# --- 3. SIDEBAR: SETTINGS & OPTIMIZATION ---
# --- 3. SIDEBAR: STATION MASTER & METADATA ---
# Hardcoded database for Luangwa Basin Stations
STATION_DB = {
    "KKIA": {"lat": -15.33, "elev": 1152.0},
    "UNZA": {"lat": -15.39, "elev": 1279.0},
    "CHADIZA": {"lat": -14.07, "elev": 1050.0},
    "KABWE": {"lat": -14.45, "elev": 1181.0},
    "SERENJE": {"lat": -13.23, "elev": 1384.0},
    "CUSTOM": {"lat": -13.50, "elev": 900.0}
}

with st.sidebar:
    st.header("⚙️ Station Manager")
    
    # 1. Ingest Station Name
    station_name = st.selectbox("Select Station Profile", list(STATION_DB.keys()), index=5)
    
    # 2. Link Name to Lat/Elev (Auto-fill)
    default_lat = STATION_DB[station_name]["lat"]
    default_elev = STATION_DB[station_name]["elev"]
    
    # Allow manual override if "CUSTOM" is selected
    lat = st.number_input("Latitude", value=default_lat, format="%.2f")
    elev = st.number_input("Altitude (m)", value=default_elev, format="%.1f")
    
    st.info(f"🚨 Active: {station_name} ({lat}, {elev}m)")
    
    st.header("🎛️ Calibration")
    kc = st.slider("Kc (Crop Coeff)", 0.1, 1.5, 0.80)
    ks = st.slider("Ks (Stress Coeff)", 0.1, 1.0, 1.00)
    phase_shift = st.slider("Spearman Phase Shift", -6, 6, 0)
    
    st.header("🚀 Research Optimizers")
    if st.button("Optimize Spearman (Cross-Correlation)"):
        # pass # Optimization logic here
            # Inside the "Optimize Spearman" button block:
        if 'AA' in st.session_state and 'SetE' in st.session_state:
            # Get a representative ground series and your calibrated satellite series
            g_ref = list(st.session_state['AA'].values())[0]
            s_ref = st.session_state['SetE']
            
            best_rho = -1
            best_shift = 0
            
            for shift in range(-6, 7): # Testing -6 to +6 months
                s_shifted = s_ref.shift(shift)
                combined = pd.concat([g_ref, s_shifted], axis=1, join='inner').dropna()
                if len(combined) > 5:
                    rho, _ = spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
                    if abs(rho) > best_rho:
                        best_rho = abs(rho)
                        best_shift = shift
            
            # --- THE KEY FIX ---
            st.session_state['phase_shift'] = best_shift
            st.success(f"Optimal Phase Shift: {best_shift} months")
            st.rerun()

    if st.button("Optimize Kc (Kamble Method)"):
        # pass
            # Inside the "Optimize Kc" button block:
        if 'SetE' in st.session_state:
            ndvi_series = st.session_state['SetE'] # Assuming SetE is NDVI
            # Kamble linear regression coefficients
            kc_optimized = (1.457 * ndvi_series) - 0.1725
            
            # Clip values to realistic hydrological bounds [0.1, 1.3]
            kc_optimized = kc_optimized.clip(0.1, 1.3)
            
            st.session_state['opt_kc'] = kc_optimized.mean()
            st.success(f"Optimized Kc (Kamble): {st.session_state['opt_kc']:.2f}")


    if st.button("Optimize Ks (LM Algorithm)"):
        if 'AA' in st.session_state and 'SetE' in st.session_state:
            g_raw = list(st.session_state['AA'].values())[0]
            s_raw = st.session_state['SetE']
            combined = pd.concat([g_raw, s_raw], axis=1, join='inner').dropna()
            y_true = combined.iloc[:, 0].values
            sat_eto = combined.iloc[:, 1].values

            def residuals(ks_val, eto, ground):
                return (eto * kc * ks_val) - ground

            res = least_squares(residuals, x0=[1.0], bounds=(0.1, 1.0), args=(sat_eto, y_true))
            
            # --- THE KEY FIX ---
            st.session_state['ks'] = float(res.x[0])
            st.success(f"Optimized Ks: {st.session_state['ks']:.3f}")
            st.rerun()

# --- 4. DATA UTILITIES ---
def clean_data(df, threshold=12):
    """
    Stabilized cleaning logic: Works for both Series and DataFrames.
    Step 14: Filling blanks and removing outliers.
    """
    # Force input into a DataFrame if it's a Series
    is_series = isinstance(df, pd.Series)
    if is_series:
        df = df.to_frame()

    # 1. Fill blanks (Climatological Mean)
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if 0 < missing_count < threshold:
            # Step 14.j.i: Use climatological average for gaps < 12 months
            df[col] = df[col].fillna(df.groupby(df.index.month)[col].transform('mean'))
            
    # 2. Check for Outliers (Modified Z-score / Robust MAD)
    # Citable: Iglewicz and Hoaglin (1993)
    for col in df.columns:
        median = df[col].median()
        mad = (df[col] - median).abs().median()
        if mad != 0:
            modified_z = 0.6745 * (df[col] - median) / mad
            # Replace outliers with median
            df.loc[modified_z.abs() > 3.5, col] = median
            
    # Return to original format if it was a Series
    return df.iloc[:, 0] if is_series else df

def get_300dpi_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, transparent=True)
    return buf.getvalue()

# --- 5. TABS ---
tabs = st.tabs(["🏠 Ground Data", "🛰️ Satellite EO", "⚖️ Validation A", "📊 Validation B", "📈 Statistics"])

with tabs[0]:
    st.header("Ground Station Ingest")
    files = st.file_uploader("Upload Meteo CSVs", accept_multiple_files=True)
    if files:
        for f in files:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = clean_data(df)
            
            # Step 5-6: Penman-Monteith
            eto = pyet.pm_fao56(tmean=df['Tavg'], wind=df['WindSpeed'], rs=df['SolarRad'], 
                               tmax=df['Tmax'], tmin=df['Tmin'], rh=df['RH'], 
                               elevation=elev, lat=lat)
            actual_et = eto * kc * ks
            st.session_state['AA'][f.name] = actual_et
            
            # Step 9: Visualization
            fig, ax = plt.subplots()
            ax.plot(actual_et, color='#00b300', lw=2)
            ax.set_title(f"Actual E: {f.name}", color='black')
            st.pyplot(fig)
            st.download_button(f"Download {f.name} PNG", get_300dpi_png(fig), f"{f.name}.png")

with tabs[4]:
    st.header("Statistical Validation")
    # Slot for ANOVA, Spearman, Pearson logic with #00b300 series
# import sklearn.linear_model as lm

def calibrate_sensors(s2_data, l8_data):
    """Step 11: Calibrate S2 using L8 with Duplicate Handling."""
    
    # 1. Deduplicate: If multiple rows exist for the same date, take the mean
    # This prevents the "duplicate labels" ValueError
    s2_clean = s2_data.groupby(s2_data.index).mean()
    l8_clean = l8_data.groupby(l8_data.index).mean()
    
    # 2. Strict Alignment
    # join='inner' ensures we only keep dates present in BOTH sensors
    combined = pd.concat([s2_clean, l8_clean], axis=1, join='inner').dropna()
    combined.columns = ['S2', 'L8']
    
    if len(combined) < 3:
        st.warning("⚠️ Insufficient unique overlapping dates for calibration.")
        return s2_data 
    
    # 3. Train OLS model
    X = combined['L8'].values.reshape(-1, 1)
    y = combined['S2'].values
    model = lm.LinearRegression()
    model.fit(X, y)
    
    # 4. Predict for 2013-2014
    gap_data = l8_clean[(l8_clean.index.year >= 2013) & (l8_clean.index.year <= 2014)]
    
    if not gap_data.empty:
        predictions = model.predict(gap_data.values.reshape(-1, 1))
        gap_series = pd.Series(predictions, index=gap_data.index)
        
        # 5. Create Set E (Complete Sentinel 2)
        set_e = pd.concat([gap_series, s2_clean]).sort_index()
        # Final safety check for duplicates
        set_e = set_e.groupby(set_e.index).mean()
        return set_e
    else:
        st.info("ℹ️ No 2013-2014 Landsat 8 data found.")
        return s2_clean# --- Tab 2: Satellite Earth Observation Data ---
    
with tabs[1]:
    st.header("Satellite Data Ingest & Calibration")
    
    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Upload Set A (Sentinel 2 NDVI/NDII)", accept_multiple_files=True)
        file_b = st.file_uploader("Upload Set B (Landsat 8 NDVI/NDII)", accept_multiple_files=True)
    with col2:
        file_c = st.file_uploader("Upload Set C (Landsat 8 LST)", accept_multiple_files=True)
        file_d = st.file_uploader("Upload Set D (ET Products: GLDAS, MODIS, etc.)", accept_multiple_files=True)

    if file_a and file_b:
        # Load and clean Data
        s2 = pd.read_csv(file_a[0], index_col=0, parse_dates=True).iloc[:, 0]
        l8 = pd.read_csv(file_b[0], index_col=0, parse_dates=True).iloc[:, 0]
        
        # Step 11: Calibration
        st.info("🔄 Calibrating Sentinel 2 using Landsat 8 (2013-2014)...")
        set_e = calibrate_sensors(s2, l8)
        
        # Step 14: Cleaning Set E (12-month rule)
        set_e = clean_data(set_e, threshold=12)
        st.session_state['SetE'] = set_e
        st.session_state['AB']['SetE'] = set_e # Store in session memory AB
        
        # Step 17: Visualizing Set E
        fig_e, ax_e = plt.subplots(figsize=(10, 4))
        ax_e.plot(set_e, color='#0066cc', lw=1.5) # Forced Satellite Color
        ax_e.set_title("Complete Sentinel 2 (Calibrated)", fontsize=12, color='black')
        ax_e.set_ylabel("NDVI")
        
        # Transparent 300DPI PNG Export
        st.pyplot(fig_e)
        st.download_button("📥 Download Calibrated Sentinel 2 PNG?", get_300dpi_png(fig_e), "SetE_Complete.png")
        st.download_button("📥 Download Calibrated Sentinel 2 PNG CSV", set_e.to_csv(), "SetE_Data.csv")

    # if file_d:
    #     # Load Set D products into session memory
    #     st.session_state['AB']['SetD'] = {f.name: pd.read_csv(f, index_col=0, parse_dates=True) for f in file_d}
    #     st.success("Set D Ingested.")
    # --- Tab 2: Satellite EO ---
if file_d:
    # Clean filenames by removing .csv during ingest
    st.session_state['AB']['SetD'] = {
        f.name.replace(".csv", ""): pd.read_csv(f, index_col=0, parse_dates=True) 
        for f in file_d
    }
    st.success("Set D Ingested.")    

def validate_and_rank_et(set_d_dict, proxy_data, set_label):
    """
    Step 18-19: Validates Set D against a proxy (Set E or Set C) 
    and ranks the Top 3 members.
    """
    results = []
    
    # Standardize proxy (Deduplicate and clean)
    proxy = proxy_data.groupby(proxy_data.index).mean().dropna()

    for name, df_prod in set_d_dict.items():
        # Ensure product is a Series and aligned with proxy
        prod_series = df_prod.iloc[:, 0].groupby(df_prod.index).mean()
        
        # Align dates
        combined = pd.concat([prod_series, proxy], axis=1, join='inner').dropna()
        combined.columns = ['ET_Prod', 'Proxy']
        
        if len(combined) > 5:
            # Step 26: Spearman's Rho
            rho, _ = spearmanr(combined['ET_Prod'], combined['Proxy'])
            results.append({"Product": name, "Rho": abs(rho), "Data": prod_series})

    # Step 19: Rank and pick Top 3
    # Sort by absolute Rho (best fit)
    ranked_results = sorted(results, key=lambda x: x['Rho'], reverse=True)[:25]
    
    # Create the resulting set (F or G)
    top_3_dict = {res['Product']: res['Data'] for res in ranked_results}
    
    return top_3_dict, ranked_results

# --- Integration into Tab 3: Validation A ---
with tabs[2]:
    st.header("⚖️ Validation A: ET Products vs Proxies (Step 18-20)")
    
    if 'SetE' in st.session_state and 'AB' in st.session_state and 'SetD' in st.session_state['AB']:
        if st.button("⚖️ Run spacial validation:best to least"):
            set_d = st.session_state['AB']['SetD']
            set_e = st.session_state['SetE']
            set_c = st.session_state.get('SetC') # Assuming Set C was uploaded
            
            # 1. Set D vs Set E (NDVI) -> Set F
            st.subheader("📊 Ranking Set F (Sentinel 2 NDVI Validation)")
            set_f, rank_f = validate_and_rank_et(set_d, set_e, "F")
            st.session_state['SetF'] = set_f
            
            for i, res in enumerate(rank_f):
                st.write(f"Rank {i+1}: **{res['Product']}** (Rho: {res['Rho']:.3f})")

            # 2. Set D vs Set C (LST) -> Set G
            if set_c is not None:
                st.subheader("📊 Ranking Set G (Landsat 8 LST Validation)")
                set_g, rank_g = validate_and_rank_et(set_d, set_c, "G")
                st.session_state['SetG'] = set_g
                for i, res in enumerate(rank_g):
                    st.write(f"Rank {i+1}: **{res['Product']}** (Rho: {res['Rho']:.3f})")
            
            # Step 20: Consolidate to session memory AB
            st.session_state['AB']['SetF'] = set_f
            st.session_state['AB']['SetG'] = set_g if set_c is not None else {}
            st.success("Validation A Complete. Set F and Set G stored in Memory AB.")
            
            # Step 17 Logic for Set F visualization (Blue Series)
            fig_f, ax_f = plt.subplots(figsize=(10, 4))
            for name, data in set_f.items():
                ax_f.plot(data, label=name, lw=1.5)
            ax_f.set_title("Top 5 ET Products (NDVI Validated)", color='black')
            ax_f.legend()
            st.pyplot(fig_f)
            st.download_button("📥 Download Set F PNG", get_300dpi_png(fig_f), "SetF_Validation.png")
                    
# --- Tab 4: Validation B (Ground vs. Satellite Winners) ---
with tabs[3]:
    st.header("📊 Validation B: Ground (AA) vs. Satellite Winners (AB)")

    # Ensure AA and SetF exist before rendering selectors
    if 'AA' in st.session_state and 'SetF' in st.session_state and st.session_state['AA'] and st.session_state['SetF']:
        
        col_a, col_b = st.columns(2)
        with col_a:
            ground_choice = st.selectbox("Choose Ground Station (AA):", list(st.session_state['AA'].keys()))
        with col_b:
            # Combine winners from Set F and Set G for a comprehensive list
            winners_f = list(st.session_state['SetF'].keys())
            winners_g = list(st.session_state['SetG'].keys()) if 'SetG' in st.session_state else []
            full_winner_list = list(set(winners_f + winners_g))
            sat_choice = st.selectbox("Choose Satellite Winner (AB):", full_winner_list)

        if st.button("⚖️ Run Final Validation"):
            # 1. Fetch & Resample
            g_raw = st.session_state['AA'][ground_choice]
            g_monthly = g_raw.to_frame().resample('MS').mean()

            s_raw = st.session_state['SetF'].get(sat_choice, st.session_state.get('SetG', {}).get(sat_choice))
            s_monthly = s_raw.to_frame().resample('MS').mean()

            # --- APPLY CALIBRATION PARAMETERS ---
            # Apply the Phase Shift (Slider or Optimized value)
            if phase_shift != 0:
                s_monthly = s_monthly.shift(phase_shift)
                st.info(f"🕒 Satellite series shifted by {phase_shift} months for temporal alignment.")

            # 2. Align and Validate
            v_aligned = pd.concat([g_monthly, s_monthly], axis=1, join='inner').dropna()
            v_aligned.columns = ['Ground', 'Satellite']

            v_aligned['Satellite'] = v_aligned['Satellite']* kc *ks
            
            if not v_aligned.empty:
                v_aligned.columns = ['Ground', 'Satellite']
                # We insert the AD Test for normality
                # --- NEW: ANDERSON-DARLING NORMALITY TEST ---
                st.subheader("🧪 Distribution Diagnostic (Anderson-Darling)")
                ad_cols = st.columns(2)
                normality_results = {}


                # ... [Existing A-D Stat Loop] ...
                for i, col in enumerate(['Ground', 'Satellite']):
                    res = stats.anderson(v_aligned[col], dist='norm')
                    is_normal = res.statistic < res.critical_values[2] 
                    normality_results[col] = is_normal
                    skew_val = v_aligned[col].skew()
                    kurt_val = v_aligned[col].kurtosis()
                    
                    with ad_cols[i]:
                        status = "✅ Normal" if is_normal else "❌ Non-Normal"
                        st.markdown(f"**{col} Distribution:** {status}")
                        st.caption(f"A-D Stat: {res.statistic:.3f} (Crit: {res.critical_values[2]:.3f})") #*

                        # 2. Display the new "Shape" metrics
                        # Formatting to 2 decimal places for scientific clarity
                        st.write(f"**Skewness:** {skew_val:.2f}")
                        st.write(f"**Kurtosis:** {kurt_val:.2f}")

                        # Add a quick interpretation tooltip
                        if abs(skew_val) > 1:
                            st.caption("🚨 Highly Skewed Data")
                        elif abs(skew_val) > 0.5:
                            st.caption("⚠️ Moderately Skewed")
                        else:
                            st.caption("✔️ Fairly Symmetrical")


                # --- NEW: Q-Q PLOT CHARTING CODE ---
                # --- Q-Q PLOT CHARTING CODE ---
                st.write("#### 📉 Probability Plots (Q-Q Diagnostics)")
                fig_qq, qq_axes = plt.subplots(1, 2, figsize=(12, 4))
                
                for i, col in enumerate(['Ground', 'Satellite']):
                    # This generates the visual "line" for the A-D test
                    stats.probplot(v_aligned[col], dist="norm", plot=qq_axes[i])
                    qq_axes[i].set_title(f"Q-Q Plot: {col}")
                    
                    # Apply your thesis styling (Black borders)
                    for spine in qq_axes[i].spines.values():
                        spine.set_edgecolor('black')
                
                plt.tight_layout()
                st.pyplot(fig_qq)

                # --- NEW: PNG & CSV DOWNLOAD BUTTONS ---
                q_col1, q_col2 = st.columns(2)
                
                with q_col1:
                    st.download_button(
                        label="📸 Download Q-Q Plot (300 DPI)",
                        data=get_300dpi_png(fig_qq),
                        file_name=f"QQ_Plot_{ground_choice}_vs_{sat_choice}.png",
                        mime="image/png"
                    )
                
                with q_col2:
                    st.download_button(
                        label="📥 Download Q-Q Raw Data (CSV)",
                        data=v_aligned[['Ground', 'Satellite']].to_csv(index=True),
                        file_name=f"QQ_Data_{ground_choice}_vs_{sat_choice}.csv",
                        mime="text/csv"
                    )

                # --- ADDED: DOWNLOAD BUTTON FOR QQ PLOT ---
                        
                if not normality_results['Ground'] or not normality_results['Satellite']:
                    st.warning("💡 **Recommendation:** One or both datasets are Non-Normal. Use **Spearman ρ** for correlation and **Kruskal-Wallis** for variance.")
                else:
                    st.success("💡 **Recommendation:** Both datasets are Normal. **Pearson r** and **ANOVA** are statistically valid.")
            
                
                # --- STATISTICS CALCULATION ---
                rho, _ = spearmanr(v_aligned['Ground'], v_aligned['Satellite'])
                r_val, _ = pearsonr(v_aligned['Ground'], v_aligned['Satellite'])
                r2_lin = r2_score(v_aligned['Ground'], v_aligned['Satellite'])
                kw_stat, p_kw = stats.kruskal(v_aligned['Ground'], v_aligned['Satellite'])

                # Save results for Tab 4 use
                st.session_state['AC'][f"{ground_choice}_vs_{sat_choice}"] = v_aligned

                # --- DYNAMIC METRICS DISPLAY ---
                st.write("### 📏 Statistical Metrics")
                m1, m2, m3, m4 = st.columns(4)
                
                
                is_normal = normality_results['Ground'] and normality_results['Satellite']
                
                m1.metric("Spearman ρ", f"{rho:.3f}", help="Best for Non-Normal data")
                m2.metric("Pearson r", f"{r_val:.3f}", delta="Normal" if is_normal else None)
                m3.metric("Kruskal p-value", f"{p_kw:.4f}")
                m4.metric("Linear R²", f"{r2_lin:.3f}")

                if p_kw < 0.05:
                    st.error(f"⚠️ **Kruskal-Wallis Warning:** Significant difference in medians (p={p_kw:.4f}). Satellite may be biased.")
                else:
                    st.success(f"✅ **Kruskal-Wallis Pass:** No significant difference in medians (p={p_kw:.4f}).")

                
                
                # --- CHARTING (Spearman Visualization) ---
                st.write("#### 🛰️ Correlation Scatter Plot")
                fig_scat, ax_scat = plt.subplots(figsize=(6, 5))
                ax_scat.scatter(v_aligned['Ground'], v_aligned['Satellite'], alpha=0.5, color='#0066cc')
                
                # Add 1:1 line (The goal)
                lims = [np.min([ax_scat.get_xlim(), ax_scat.get_ylim()]), 
                        np.max([ax_scat.get_xlim(), ax_scat.get_ylim()])]
                ax_scat.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label="1:1 Perfect Match")
                
                ax_scat.set_xlabel("Ground ET (mm/day)", family='serif')
                ax_scat.set_ylabel("Satellite ET (mm/day)", family='serif')
                ax_scat.legend(prop={'family': 'serif'})
                
                # Dark Borders for Thesis Quality
                for spine in ax_scat.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.5)

                st.pyplot(fig_scat)

                # --- NEW: PNG & CSV DOWNLOAD BUTTONS ---
                scat_col1, scat_col2 = st.columns(2)
                
                with scat_col1:
                    st.download_button(
                        label="📸 Download Scatter Plot (300 DPI)",
                        data=get_300dpi_png(fig_scat),
                        file_name=f"Correlation_Scatter_{ground_choice}.png",
                        mime="image/png"
                    )
                
                with scat_col2:
                    st.download_button(
                        label="📥 Download Scatter Data (CSV)",
                        data=v_aligned[['Ground', 'Satellite']].to_csv(index=True),
                        file_name=f"Correlation_Data_{ground_choice}.csv",
                        mime="text/csv"
                    )

                # --- POLYNOMIAL R2 & EQUATION ---
                # x = Ground, y = Satellite
                weights = np.polyfit(v_aligned['Ground'], v_aligned['Satellite'], 2)
                model_poly = np.poly1d(weights)
                y_pred_poly = model_poly(v_aligned['Ground'])
                r2_poly = r2_score(v_aligned['Satellite'], y_pred_poly)             
                
                # Create the equation string: y = ax² + bx + c
                poly_eqn = f"y = {weights[0]:.4f}x² + {weights[1]:.4f}x + {weights[2]:.4f}"

                # 2. Rendering the View
                st.write("#### 📈 Polynomial Regression Analysis")
                fig_poly, ax_poly = plt.subplots(figsize=(8, 5))

                # Scatter points of actual data
                ax_poly.scatter(v_aligned['Ground'], v_aligned['Satellite'], alpha=0.4, color='#0066cc', label="Observed Data")

                # Create smooth curve for the polynomial line
                x_smooth = np.linspace(v_aligned['Ground'].min(), v_aligned['Ground'].max(), 100)
                ax_poly.plot(x_smooth, model_poly(x_smooth), color='red', lw=2.5, label=f"2nd Deg Polynomial (R²={r2_poly:.3f})")

                # Add 1:1 Reference Line for perspective
                lims = [v_aligned['Ground'].min(), v_aligned['Ground'].max()]
                ax_poly.plot(lims, lims, 'k--', alpha=0.5, label="1:1 Perfect Match")

                # Formatting for Thesis
                ax_poly.set_title(f"Polynomial Fit: {ground_choice} vs {sat_choice}", family='serif', fontsize=12)
                ax_poly.set_xlabel("Ground ET (mm/day)", family='serif')
                ax_poly.set_ylabel("Satellite ET (mm/day)", family='serif')
                ax_poly.legend(prop={'family': 'serif'})

                # Display the equation as a caption
                st.info(f"**Regression Equation:** `{poly_eqn}`")

                # 3. View the Chart
                st.pyplot(fig_poly)

                # 4. Download the Chart
                st.download_button(
                    label="📸 Download Polynomial Chart (300 DPI)",
                    data=get_300dpi_png(fig_poly),
                    file_name=f"Polynomial_Fit_{ground_choice}.png",
                    mime="image/png"
                )

                

                # --- CHARTING ---
                fig_v, ax_v = plt.subplots(figsize=(10, 5))
                ax_v.plot(v_aligned.index, v_aligned['Ground'], color='#00b300', label="Ground")
                ax_v.plot(v_aligned.index, v_aligned['Satellite'], color='#0066cc', ls='--', label="Satellite")
                ax_v.legend()
                st.pyplot(fig_v)

                
            
                # Statistics (Steps 26-30)
                rho, _ = spearmanr(v_aligned['Ground'], v_aligned['Satellite'])
                r_val, _ = pearsonr(v_aligned['Ground'], v_aligned['Satellite'])
                tau, _ = kendalltau(v_aligned['Ground'], v_aligned['Satellite'])
                r2_lin = r2_score(v_aligned['Ground'], v_aligned['Satellite'])
                
                # Step 21: Save results to Session Memory (AC)
                st.session_state['AC'][f"{ground_choice}_vs_{sat_choice}"] = v_aligned
                
                
                # Step 25: Visualization (Ground vs Satellite)
                # Forced Times New Roman and Black Annotations
                fig_v, ax_v = plt.subplots(figsize=(10, 5))
                ax_v.plot(v_aligned.index, v_aligned['Ground'], color='#00b300', lw=2, label=f"Ground: {ground_choice}")
                ax_v.plot(v_aligned.index, v_aligned['Satellite'], color='#0066cc', lw=1.5, ls='--', label=f"Sat: {sat_choice}")
                
                ax_v.set_title(f"Satellite data vs Met Station data: {ground_choice} vs {sat_choice}", color='black', fontsize=14, family='serif')
                ax_v.set_ylabel("Evaporation (mm/day)", color='black', family='serif')
                ax_v.legend(prop={'family': 'serif'})
                
                # Dark Borders (Step 24.q)
                for spine in ax_v.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.5)

                st.pyplot(fig_v)

                # Step 24: CSV & PNG Downloads
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("📥 Download Validation CSV", v_aligned.to_csv(), f"Validation_{ground_choice}.csv")
                with c2:
                    st.download_button("📸 Download 300DPI PNG", get_300dpi_png(fig_v), "Validation_B.png")

                # --- ADD ANOVA CALCULATION ---
                # f_oneway is the standard ANOVA test for comparing means
                f_stat, p_anova = stats.f_oneway(v_aligned['Ground'], v_aligned['Satellite'])

                st.write("#### 🎻 Distribution Comparison (Violin Plot)")
                fig_violin, ax_violin = plt.subplots(figsize=(8, 5))

                # Creating the violin plot
                # showmeans=True adds a point for the average ET
                parts = ax_violin.violinplot([v_aligned['Ground'], v_aligned['Satellite']], 
                                            showmeans=True, showmedians=True)

                # Styling the 'Violins' for Ground (Green) and Satellite (Blue)
                colors = ['#00b300', '#0066cc']
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[i])
                    pc.set_edgecolor('black')
                    pc.set_alpha(0.6)

                # Formatting the axes
                ax_violin.set_xticks([1, 2])
                ax_violin.set_xticklabels(['Ground', 'Satellite'], family='serif')
                ax_violin.set_ylabel("Evaporation (mm/day)", family='serif')
                ax_violin.set_title(f"Data Density: {ground_choice} vs {sat_choice}", family='serif')

                # Standardizing borders for academic publication
                for spine in ax_violin.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.2)

                st.pyplot(fig_violin)

                # --- DOWNLOAD BUTTONS ---
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.download_button(
                        label="📸 Download Violin Plot (300 DPI)",
                        data=get_300dpi_png(fig_violin),
                        file_name=f"ViolinPlot_{ground_choice}.png",
                        mime="image/png"
                    )
                with v_col2:
                    st.download_button(
                        label="📥 Download Plot Data (CSV)",
                        data=v_aligned[['Ground', 'Satellite']].to_csv(index=True),
                        file_name=f"Violin_Data_{ground_choice}.csv",
                        mime="text/csv"
                    )

                st.write("#### 🌊 Distribution Overlap (KDE Plot)")
                fig_kde, ax_kde = plt.subplots(figsize=(8, 4))

                # Plotting the density curves
                v_aligned['Ground'].plot(kind='kde', ax=ax_kde, label='Ground', color='green', lw=2)
                v_aligned['Satellite'].plot(kind='kde', ax=ax_kde, label='Satellite', color='blue', lw=2, ls='--')

                ax_kde.set_xlabel("Evaporation (mm/day)", family='serif')
                ax_kde.set_ylabel("Density", family='serif')
                ax_kde.set_title(f"Density Distribution: {ground_choice} vs {sat_choice}", family='serif')
                ax_kde.legend()

                for spine in ax_kde.spines.values():
                    spine.set_edgecolor('black')

                st.pyplot(fig_kde)

                # --- DOWNLOAD SECTION ---
                # Create two columns so the buttons sit side-by-side
                dl_col1, dl_col2 = st.columns(2)

                with dl_col1:
                    st.download_button(
                        label="📸 Download KDE Plot (300 DPI)",
                        data=get_300dpi_png(fig_kde),
                        file_name=f"KDE_Plot_{ground_choice}.png",
                        mime="image/png"
                    )

                with dl_col2:
                    st.download_button(
                        label="📥 Download KDE Raw Data (CSV)",
                        data=v_aligned.to_csv(index=True),
                        file_name=f"KDE_Data_{ground_choice}.csv",
                        mime="text/csv"
                    )


                # 1. CALCULATE (The math must happen first)
                residuals = v_aligned['Satellite'] - v_aligned['Ground']

                # 2. DEFINE THE FIGURE (This creates the variable 'fig_res')
                st.write("#### 🔍 Residual Analysis (Error Distribution)")
                fig_res, ax_res = plt.subplots(figsize=(10, 4))

                # 3. PLOT THE DATA
                ax_res.scatter(v_aligned['Ground'], residuals, color='purple', alpha=0.5)
                ax_res.axhline(y=0, color='black', linestyle='--', lw=2)
                ax_res.set_title("Residual Plot: Ground ET vs Error", family='serif')
                ax_res.set_xlabel("Observed Ground ET (mm/day)", family='serif')
                ax_res.set_ylabel("Residual (Sat - Ground)", family='serif')

                for spine in ax_res.spines.values():
                    spine.set_edgecolor('black')

                # 4. DISPLAY THE CHART
                st.pyplot(fig_res)

                # 5. DOWNLOAD (Now 'fig_res' exists and is ready)
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.download_button(
                        label="📸 Download Residual Plot (300 DPI)",
                        data=get_300dpi_png(fig_res),  # Now this will work!
                        file_name=f"Residual_Scatter_{ground_choice}.png",
                        mime="image/png"
                    )
                with res_col2:
                    st.download_button(
                        label="📥 Download Residual Data (CSV)",
                        data=pd.DataFrame({'Ground_ET': v_aligned['Ground'], 'Residual': residuals}).to_csv(index=True),
                        file_name=f"Residual_Data_{ground_choice}.csv",
                        mime="text/csv"
                    )

                st.write("#### ⚖️ Error Frequency (Residual Distribution)")
                fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
                
                # Plotting the histogram of residuals
                # bins=15 is usually a good balance for monthly data
                count, bins, ignored = ax_hist.hist(residuals, bins=15, color='purple', alpha=0.6, edgecolor='black', density=True)
                
                # Add a "Best Fit" normal line to the error distribution
                mu_res, std_res = stats.norm.fit(residuals)
                p_fit = stats.norm.pdf(bins, mu_res, std_res)
                ax_hist.plot(bins, p_fit, 'k', linewidth=2, label=f'Normal Fit (μ={mu_res:.2f})')
                
                ax_hist.axvline(0, color='red', linestyle='--', label='Zero Error Line')
                ax_hist.set_xlabel("Residual Magnitude (mm/day)")
                ax_hist.set_ylabel("Frequency")
                ax_hist.legend()
                
                st.pyplot(fig_hist)

                # --- NEW: PNG & CSV DOWNLOAD BUTTONS FOR HISTOGRAM ---
                h_col1, h_col2 = st.columns(2)
                
                with h_col1:
                    st.download_button(
                        label="📸 Download Histogram (300 DPI)",
                        data=get_300dpi_png(fig_hist),
                        file_name=f"Residual_Histogram_{ground_choice}.png",
                        mime="image/png"
                    )
                
                with h_col2:
                    # Create a simple DataFrame of the residuals for CSV export
                    df_res_export = pd.DataFrame({
                        'Date': v_aligned.index,
                        'Residual_Error': residuals
                    })
                    st.download_button(
                        label="📥 Download Residual List (CSV)",
                        data=df_res_export.to_csv(index=False),
                        file_name=f"Residual_Values_{ground_choice}.csv",
                        mime="text/csv"
                    )


                # Ensure all other variables are also calculated here
                rho, _ = spearmanr(v_aligned['Ground'], v_aligned['Satellite'])
                r_val, _ = pearsonr(v_aligned['Ground'], v_aligned['Satellite'])
                tau, _ = kendalltau(v_aligned['Ground'], v_aligned['Satellite'])
                r2_lin = r2_score(v_aligned['Ground'], v_aligned['Satellite'])
                kw_stat, p_kw = stats.kruskal(v_aligned['Ground'], v_aligned['Satellite'])

                

                # --- RESIDUAL CALCULATION ---
                # Residuals = Predicted (Satellite) - Observed (Ground)
                residuals = v_aligned['Satellite'] - v_aligned['Ground']
                rmse = np.sqrt(np.mean(residuals**2))
                bias = np.mean(residuals)

                # --- UPDATED MASTER METRICS TABLE ---
                summary_data = {
                    "Metric": [
                        "Ground Skewness", "Ground Kurtosis", 
                        "A-D Normality (Ground)", "A-D Normality (Sat)",
                        "Spearman Rho (ρ)", "Pearson r", 
                        "Kendall Tau (τ)", "Linear R²",
                        "Polynomial R² (2nd Deg)", # <--- ADDED
                        "ANOVA p-value", "Kruskal-Wallis p-value", "RMSE (mm/day)", "Mean Bias (mm/day)"
                    ],
                    "Value": [
                        round(v_aligned['Ground'].skew(), 3), 
                        round(v_aligned['Ground'].kurtosis(), 3),
                        "Normal" if normality_results['Ground'] else "Non-Normal",
                        "Normal" if normality_results['Satellite'] else "Non-Normal",
                        round(rho, 3), 
                        round(r_val, 3), 
                        round(tau, 3), 
                        round(r2_lin, 3),
                        round(r2_poly, 3),        # <--- ADDED
                        round(p_anova, 4), 
                        round(p_kw, 4),
                        round(rmse, 3),
                        round(bias, 3)
                    ],
                    "Note": [
                        "Shape", "Shape", "Distribution", "Distribution",
                        "Correlation", "Linearity", "Robustness", "Linear Fit",
                        poly_eqn,                 # <--- ADDED (Shows the equation here!)
                        "Mean Equality", "Median Equality", "Average Error Magnitude", "Systematic Over/Under estimation"
                    ]
                }

                df_summary = pd.DataFrame(summary_data)
                st.table(df_summary) # st.table makes it look cleaner for reports

                # Download Button for the Table
                st.download_button(
                    label="📥 Download Master Statistics CSV",
                    data=df_summary.to_csv(index=False),
                    file_name=f"Master_Stats_{ground_choice}_vs_{sat_choice}.csv",
                    mime="text/csv"
                )    

            else:
                st.error("❌ No overlapping dates found.")
                st.info(f"Ground dates: {g_monthly.index.min()} to {g_monthly.index.max()}")
                st.info(f"Satellite dates: {s_monthly.index.min()} to {s_monthly.index.max()}")
    else:
        st.warning("⚠️ Please complete Ground Ingest (AA) and Validation A (Set F) first.")

with tabs[4]:
    st.header("📈 Advanced Statistical Validation Hub")

    # GATE 1: Check if Validation B has been run and data exists in session state
    if 'AC' in st.session_state and st.session_state['AC']:
        comparison_key = st.selectbox("Select Validation Pair:", list(st.session_state['AC'].keys()))
        data = st.session_state['AC'][comparison_key]
        
        # 1. RUN CALCULATIONS (Variables are defined locally here)
        rho, _ = spearmanr(data['Ground'], data['Satellite'])
        r_pea, _ = pearsonr(data['Ground'], data['Satellite'])
        tau, _ = kendalltau(data['Ground'], data['Satellite'])
        r2_lin = r2_score(data['Ground'], data['Satellite'])
        
        # ANOVA calculation
        f_stat, p_anova = stats.f_oneway(data['Ground'], data['Satellite']) 

        # 2. PERFORMANCE GRADING 
        st.subheader("🎓 Station Performance Grade")

        if p_anova < 0.05:
            if rho > 0.8:
                grade = "🌟 Excellent (High Fidelity)"
                desc = "The satellite product perfectly mirrors ground physics."
            elif rho > 0.6:
                grade = "✅ Good (Reliable)"
                desc = "Strong seasonal agreement with minor magnitude shifts."
            else:
                grade = "⚠️ Marginal (Bias Present)"
                desc = "Patterns align but magnitudes differ significantly."
        else:
            grade = "❌ Not Significant"
            desc = "The product does not statistically represent this station."

        # Display Interpretation Card
        st.info(f"**Result for {comparison_key}:**\n\n**Grade:** {grade}\n\n**Interpretation:** {desc}")
        
        # 3. BATCH EXPORT (Nested here so it has access to 'grade' and 'p_anova')
        st.divider()
        st.subheader("📥 Master Research Export")
        
        if st.button("📦 Generate Full Station Report"):
            buf_zip = io.BytesIO()
            with zipfile.ZipFile(buf_zip, "w") as zf:
                # Save CSV
                zf.writestr(f"{comparison_key}_data.csv", data.to_csv())
                
                # Save Summary Statistics
                stats_text = f"""
                LUANGWA MASTER LAB 2.0 - RESEARCH REPORT
                Station Pair: {comparison_key}
                ------------------------------------------
                Spearman Rho: {rho:.4f}
                Pearson r: {r_pea:.4f}
                ANOVA p-value: {p_anova:.4f}
                FINAL GRADE: {grade}
                """
                zf.writestr("summary_statistics.txt", stats_text)
                
            st.download_button(
                label="📥 Download Complete Thesis Package (ZIP)",
                data=buf_zip.getvalue(),
                file_name=f"Research_Report_{comparison_key}.zip",
                mime="application/zip"
            )

    else:
        st.info("💡 Run 'Validation B' in the previous tab to generate statistics.")