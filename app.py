import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import joblib
import folium
from streamlit_folium import st_folium

# --- Page Config ---
st.set_page_config(page_title="TrafficFlow", layout="wide", page_icon="🚦")

# --- Colors ---
MAIN_BG = "#FFD6E8"       # Lighter pink main background
INPUT_BG = "#FFFFFF"      # Sidebar white background
DARK_PURPLE = "#4B0082"   # Dark purple header
PRIMARY = "#F9A8D4"
SECONDARY = "#B084CC"
ACCENT = "#FCD34D"

sns.set_theme(style="whitegrid")  # Seaborn compatible style
plt.style.use('ggplot')

# --- CSS Styling ---
st.markdown(f"""
<style>
/* Main page background */
.stApp {{
    background-color: {MAIN_BG};
}}

/* Sidebar background and text */
.css-1d391kg {{
    background-color: {INPUT_BG} !important;
    color: black !important;
    padding: 20px;
    border-radius: 15px;
}}

/* Sidebar input text */
.css-1d391kg .st-bx {{
    color: black !important;
}}

/* Header styling with hover animation */
h1 {{
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    color: {DARK_PURPLE};
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    margin-bottom: 30px;
    transition: all 0.3s ease;
}}
h1:hover {{
    transform: scale(1.05);
    text-shadow: 2px 2px 5px rgba(0,0,0,0.2);
}}

/* Metric card animation */
.metric-card {{
    transition: transform 0.5s ease, background 0.5s ease;
}}
.metric-card:hover {{
    transform: scale(1.05);
}}

/* Metric card text color black */
.metric-card h4, .metric-card h2 {{
    color: black !important;
}}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown(f"<h1>TrafficFlow Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Load Data & Model ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/merged_traffic.csv")
    df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    if 'hour' not in gdf.columns:
        gdf['hour'] = pd.to_datetime(gdf['date_time']).dt.hour
    if 'weekday' not in gdf.columns:
        gdf['weekday'] = pd.to_datetime(gdf['date_time']).dt.weekday
    return gdf

df = load_data()
model_path = "models/traffic_model.joblib"
model = joblib.load(model_path)

# --- Sidebar ---
st.sidebar.header("🎀 Traffic Prediction Inputs")
road_type = st.sidebar.selectbox("Select Road Type 🚗", df['highway'].unique())
hour = st.sidebar.slider("Hour of Day ⏰", 0, 23, 8)
temp = st.sidebar.number_input("Temperature 🌡️ (°C)", value=25.0)
rain = st.sidebar.number_input("Rain 🌧️ (mm/h)", value=0.0)
lanes = st.sidebar.number_input("Number of Lanes 🛣️", 1, 8, 2)
speed_limit = st.sidebar.number_input("Speed Limit 🚦 (km/h)", value=50)
weekday = st.sidebar.selectbox("Weekday 📅", list(range(7)), index=0)
is_weekend = 1 if weekday >= 5 else 0

# --- Tabs ---
tabs = st.tabs(["📊 EDA", "🗺️ Traffic Map", "🔮 Prediction", "💡 Insights"])

# --- Metric Card Function ---
def metric_card(title, value, gradient_start, gradient_end):
    st.markdown(f"""
    <div class="metric-card" style="
        background: linear-gradient(to right, {gradient_start}, {gradient_end});
        padding: 20px;
        border-radius: 20px;
        margin-bottom: 15px;
        text-align:center;
    ">
        <h4>{title}</h4>
        <h2>{value}</h2>
    </div>
    """, unsafe_allow_html=True)

# --- Tab 1: EDA ---
with tabs[0]:
    st.subheader("Traffic Distribution by Road Type & Hour")
    st.markdown("Shows how traffic volume varies by hour for the selected road type.")
    df_filtered = df[df['highway'] == road_type]
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(x='hour', y='traffic_volume', data=df_filtered, palette=[PRIMARY, SECONDARY, ACCENT])
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Traffic Volume Categories")
    st.markdown("Breakdown of traffic volume into Low, Medium, High, and Very High categories.")
    df['traffic_category'] = pd.cut(df['traffic_volume'], 
                                    bins=[0,500,1000,1500,df['traffic_volume'].max()], 
                                    labels=['Low','Medium','High','Very High'])
    fig2, ax2 = plt.subplots(figsize=(8,5))
    sns.countplot(x='traffic_category', data=df, palette=[PRIMARY, SECONDARY, ACCENT])
    st.pyplot(fig2)

    st.subheader("Feature Importance")
    st.markdown("Shows which features have the most influence on traffic predictions.")
    try:
        feature_cols = ["hour","temp","rain_1h","lanes","maxspeed","weekday","is_weekend"]
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(10,6))
        sns.barplot(x=feat_imp.values, y=feat_imp.index, palette=[PRIMARY, SECONDARY, ACCENT])
        st.pyplot(fig3)
    except:
        st.info("Feature importance not available for this model.")

# --- Tab 2: Traffic Map ---
with tabs[1]:
    st.subheader("📍 Nairobi Traffic Locations")
    st.markdown("Map showing sampled traffic locations in Nairobi.")
    m = folium.Map(location=[-1.2921, 36.8219], zoom_start=12)
    for _, row in df.sample(500).iterrows():
        folium.CircleMarker(location=[row['latitude'], row['longitude']],
                            radius=3,
                            color=PRIMARY,
                            fill=True,
                            fill_color=SECONDARY,
                            fill_opacity=0.7).add_to(m)
    st_folium(m, width=700, height=500)

# --- Tab 3: Prediction ---
with tabs[2]:
    st.subheader("Predict Traffic Volume 🔮")
    st.markdown("Input traffic conditions to predict traffic volume on selected road type.")
    if st.button("Predict"):
        x_input = pd.DataFrame([{
            "hour": hour,
            "temp": temp,
            "rain_1h": rain,
            "lanes": lanes,
            "maxspeed": speed_limit,
            "weekday": weekday,
            "is_weekend": is_weekend
        }])
        pred = model.predict(x_input)[0]
        metric_card("Predicted Traffic Volume", f"{int(pred)} vehicles", SECONDARY, ACCENT)

# --- Tab 4: Insights ---
with tabs[3]:
    st.subheader("💡 Insights & Recommendations")
    st.markdown("""
    **Findings:**
    - Traffic peaks during rush hours (7-9 AM, 5-7 PM).  
    - Highways and multi-lane roads experience higher traffic volumes.  
    - Rain and bad weather slightly reduce traffic speeds.  

    **Recommendations:**
    - Consider alternate routes or signal adjustments during peak times.  
    - GPS and traffic data must be used responsibly to protect privacy.  

    **Proposal for Other Fields:**
    - Healthcare: Predict hospital patient inflow and optimize staffing.  
    - Finance: Forecast transaction volumes or cash demand.  
    - Marketing: Predict customer visits, ad engagement, or campaign impact.  
    """)
    metric_card("Average Traffic", int(df['traffic_volume'].mean()), PRIMARY, SECONDARY)
    metric_card("Maximum Traffic", int(df['traffic_volume'].max()), SECONDARY, ACCENT)
    metric_card("Minimum Traffic", int(df['traffic_volume'].min()), ACCENT, PRIMARY)
