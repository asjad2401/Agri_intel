import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import requests

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Agri-Intel: Real-Time Dashboard", layout="wide")

# REPLACE THIS WITH YOUR OPENWEATHERMAP API KEY
API_KEY = "a80d092216e475bd51118c7ea67d7dd1" 

# Load Model & Data
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("final_model.pkl")
        data = pd.read_csv("final_dataset_ml_ready.csv")
        gdf = gpd.read_file("punjab_districts_cleaned.geojson")
        return model, data, gdf
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None

model, df, gdf = load_assets()

# --- HELPER: FETCH LIVE WEATHER ---
def get_live_weather(city, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city},PK&appid={api_key}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            desc = data['weather'][0]['description']
            return temp, desc
        else:
            return None, None
    except:
        return None, None

# --- 2. LOGIC: GET DEFAULTS ---
latest_year = df['Year'].max()
district_stats = df[df['Year'] == latest_year].set_index('District')

# --- 3. SIDEBAR: INTELLIGENCE ---
st.sidebar.header("📍 Location Intelligence")
district_list = sorted(df['District'].unique())
selected_district = st.sidebar.selectbox("Select Target District", district_list)

# A. Historical Defaults
try:
    defaults = district_stats.loc[selected_district]
    def_fert = float(defaults['Fertilizer_Usage_K_Tons'])
    def_ndvi = float(defaults['Mean_NDVI'])
    def_rain = float(defaults['Total_Rainfall_mm'])
    def_temp = float(defaults['Avg_Temp_C']) # Historical Avg
    def_area = float(defaults['Area_Sown_Wheat'])
except:
    def_fert, def_ndvi, def_rain, def_temp, def_area = 50.0, 0.5, 100.0, 25.0, 100.0

# B. Real-Time Data Fetching
live_temp, live_desc = get_live_weather(selected_district, API_KEY)

if live_temp is not None:
    st.sidebar.success(f"🟢 Live Data Fetched: {live_desc.title()}")
    # Update default temperature to the CURRENT live temperature
    def_temp = live_temp
else:
    st.sidebar.warning("⚠️ Using Historical Data (Live Weather Unavailable)")

st.sidebar.divider()
st.sidebar.header("🌱 Field Parameters")

# Sliders
fertilizer = st.sidebar.slider("Fertilizer (1000 Tons)", 0.0, 300.0, def_fert)
ndvi = st.sidebar.slider("Crop Health (NDVI)", 0.0, 1.0, def_ndvi)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 2000.0, def_rain, help="Seasonal Total Rainfall")

# Temperature Slider (Auto-set to Live Temp if available)
temp = st.sidebar.slider(
    f"Temperature (°C) {'[LIVE]' if live_temp else ''}", 
    min_value=-5.0, max_value=50.0, value=def_temp
)

area_sown = st.sidebar.number_input("Area Sown (Acres)", value=def_area)

# --- 4. MAIN DASHBOARD ---
st.title("🌾 Agri-Intel: Smart Yield Optimizer")

# Display Live Weather Badge if Active
if live_temp:
    st.info(f"🌤️ **Real-Time Conditions in {selected_district}:** {live_temp}°C, {live_desc.title()}. Model inputs updated.")

col_map, col_pred = st.columns([1.5, 1])

# --- MAP SECTION ---
with col_map:
    st.subheader(f"Geospatial View: {selected_district}")
    m = folium.Map(location=[31.1704, 72.7097], zoom_start=6, tiles="CartoDB positron")
    if gdf is not None:
        target_geo = gdf[gdf['District_Name_Clean'] == selected_district]
        if not target_geo.empty:
            target_geo = target_geo[['District_Name_Clean', 'geometry']]
            folium.GeoJson(
                target_geo,
                style_function=lambda x: {'fillColor': '#228B22', 'color': 'black', 'weight': 2, 'fillOpacity': 0.5},
                tooltip=selected_district
            ).add_to(m)
    st_folium(m, width=500, height=400)
# --- COLUMN 2: PREDICTION ENGINE ---
with col_pred:
    st.subheader("Yield Prediction")
    
    # Prepare Input
    input_data = pd.DataFrame({
        'Fertilizer_Usage_K_Tons': [fertilizer],
        'Mean_NDVI': [ndvi],
        'Total_Rainfall_mm': [rainfall],
        'Avg_Temp_C': [temp],
        'Area_Sown_Wheat': [area_sown]
    })
    
    # Initialize session state for prediction if it doesn't exist
    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None

    if st.button("🚀 Analyze Yield", type="primary"):
        # Store prediction in session state so it persists
        pred_value = model.predict(input_data)[0]
        st.session_state['prediction'] = pred_value
        st.session_state['input_data'] = input_data # Store inputs too for the chart
        
    # DISPLAY RESULTS (Only if prediction exists)
    if st.session_state['prediction'] is not None:
        prediction = st.session_state['prediction']
        
        # 1. Metric Display
        st.metric(label="Predicted Output", value=f"{prediction:.2f} Tons/Acre")
        
        # 2. Contextual Comparison
        try:
            avg_hist_yield = df[df['District'] == selected_district]['Yield_Wheat_Acre'].mean()
            delta = prediction - avg_hist_yield
            st.metric(label="Vs Historical Avg", value=f"{avg_hist_yield:.2f}", delta=f"{delta:.2f}")
        except:
            pass

        # 3. Recommendations
        if prediction < avg_hist_yield:
             st.error("📉 Prediction is below historical average.")
        else:
             st.success("📈 Prediction exceeds historical average!")

# --- 5. "WHAT-IF" SCENARIO & OPTIMIZER ---
st.divider()

if st.session_state['prediction'] is not None:
    st.subheader(f"🔮 Simulation & Recommendation Engine")

    # Get data from session state
    current_inputs = st.session_state['input_data']
    prediction = st.session_state['prediction']
    current_fert = current_inputs['Fertilizer_Usage_K_Tons'].iloc[0]

    # 1. Generate Synthetic Curve
    # Scan range 0 to 300 to find absolute peak
    f_range = np.linspace(0, 300, 100)
    temp_inputs = pd.concat([current_inputs] * 100, ignore_index=True)
    temp_inputs['Fertilizer_Usage_K_Tons'] = f_range
    y_range = model.predict(temp_inputs)

    # 2. Find the OPTIMAL Point (Peak of the Curve)
    max_yield_idx = np.argmax(y_range)
    max_yield = y_range[max_yield_idx]
    optimal_fert = f_range[max_yield_idx]

    # 3. Recommendation Logic
    col_rec, col_chart = st.columns([1, 2])
    
    with col_rec:
        st.markdown("### 💡 AI Analysis")
        
        # Calculate Potential Gain
        yield_gain = max_yield - prediction
        fert_diff = optimal_fert - current_fert
        
        # Threshold: Only say "Perfect" if gain is negligible (< 0.01)
        if yield_gain > 0.01:
            st.warning(f"⚠️ Optimization Opportunity Detected")
            st.metric("Potential Max Yield", f"{max_yield:.2f} Tons/Acre", delta=f"+{yield_gain:.2f}")
            st.metric("Optimal Fertilizer", f"{optimal_fert:.0f} K-Tons", delta=f"{fert_diff:+.0f}")
            
            if fert_diff > 0:
                st.info(f"👉 **Recommendation:** Increase fertilizer by **{fert_diff:.0f} K-Tons**.")
            else:
                st.info(f"👉 **Recommendation:** Reduce fertilizer by **{abs(fert_diff):.0f} K-Tons**.")
        else:
            # Strictly optimal
            st.success("✅ **System Optimized**")
            st.markdown("Current selection delivers the **maximum possible yield** for these weather conditions.")
            st.metric("Peak Yield", f"{max_yield:.2f} Tons/Acre")

    with col_chart:
        # Plotting
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(8, 4))

        # Main Curve
        sns.lineplot(
            x=f_range, y=y_range, ax=ax, 
            color='#2ecc71', linewidth=3, label='Yield Response Curve'
        )
        ax.fill_between(f_range, y_range, alpha=0.2, color='#2ecc71')

        # User's Point (Red)
        ax.scatter([current_fert], [prediction], color='#e74c3c', s=150, zorder=5, edgecolors='white', linewidth=2, label='Your Selection')
        
        # Optimal Point (Gold Star)
        ax.scatter([optimal_fert], [max_yield], color='#f1c40f', s=200, marker='*', zorder=6, edgecolors='black', label='AI Recommended Peak')

        # Annotation for User Point
        ax.annotate(
            f'Current\n{prediction:.1f}', 
            xy=(current_fert, prediction), 
            xytext=(current_fert, prediction - (max_yield*0.1)),
            ha='center', fontsize=10, fontweight='bold', color='#e74c3c',
            arrowprops=dict(facecolor='#e74c3c', shrink=0.05)
        )

        # Annotation for Optimal Point
        ax.annotate(
            f'Peak\n{max_yield:.1f}', 
            xy=(optimal_fert, max_yield), 
            xytext=(optimal_fert, max_yield + (max_yield*0.05)),
            ha='center', fontsize=10, fontweight='bold', color='#f39c12'
        )

        ax.set_xlabel("Fertilizer Input (1000 Tons)")
        ax.set_ylabel("Yield Output")
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3)
        
        st.pyplot(fig)

else:
    st.info("👈 Select parameters and click **'Analyze Yield'** to see the Optimization Simulation.")