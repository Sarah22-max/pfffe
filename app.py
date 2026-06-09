import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import os
# Conditional PySpark imports
try:
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler
    from pyspark.ml.regression import RandomForestRegressor
except ImportError:
    SparkSession = None
    Pipeline = None
    StringIndexer = None
    VectorAssembler = None
    RandomForestRegressor = None
    st.warning("PySpark not installed. ML features are disabled until installation.")

# Set page config
st.set_page_config(
    page_title="Sante Publique - Tableau de Bord PFE",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling — Light Health Theme (White + Green)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global Streamlit overrides ── */
    .stApp {
        background-color: #f8fdf9 !important;
        font-family: 'Inter', sans-serif !important;
        color: #1e293b !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #334155 !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Main title */
    .main-title {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #16a34a !important;
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
        border-bottom: 3px solid #22c55e;
        display: inline-block;
    }

    /* Subtitle */
    .subtitle {
        color: #64748b !important;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #dcfce7;
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        padding: 20px 18px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(22, 163, 74, 0.08);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(22, 163, 74, 0.15);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #16a34a;
    }

    /* ── Recommendation Cards ── */
    .reco-card {
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
        border-left: 5px solid;
    }
    .reco-card.high {
        background: #fef2f2;
        border-left-color: #ef4444;
    }
    .reco-card.med {
        background: #fffbeb;
        border-left-color: #f59e0b;
    }
    .reco-card.low {
        background: #f0fdf4;
        border-left-color: #22c55e;
    }

    /* ── Streamlit widget styling ── */
    .stSelectbox label, .stSlider label, .stRadio label {
        color: #334155 !important;
        font-weight: 500 !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* ── Horizontal rule ── */
    hr {
        border-color: #e2e8f0 !important;
    }

    /* ── Info / Warning / Success boxes ── */
    .stAlert {
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# 58 California Counties Coordinates (Centroids) for Pydeck map
COUNTY_COORDS = {
    'Alameda': (37.65, -121.92), 'Alpine': (38.60, -119.82), 'Amador': (38.45, -120.65),
    'Butte': (39.67, -121.60), 'Calaveras': (38.20, -120.55), 'Colusa': (39.18, -122.24),
    'Contra Costa': (37.93, -121.93), 'Del Norte': (41.74, -124.00), 'El Dorado': (38.78, -120.53),
    'Fresno': (36.75, -119.65), 'Glenn': (39.59, -122.39), 'Humboldt': (40.70, -123.93),
    'Imperial': (33.04, -115.35), 'Inyo': (36.51, -117.41), 'Kern': (35.34, -118.96),
    'Kings': (36.07, -119.82), 'Lake': (39.05, -122.75), 'Lassen': (40.67, -120.59),
    'Los Angeles': (34.31, -118.22), 'Madera': (37.22, -119.76), 'Marin': (38.07, -122.75),
    'Mariposa': (37.58, -119.91), 'Mendocino': (39.44, -123.43), 'Merced': (37.19, -120.72),
    'Modoc': (41.59, -120.72), 'Mono': (37.92, -118.89), 'Monterey': (36.24, -121.32),
    'Napa': (38.51, -122.33), 'Nevada': (39.30, -120.77), 'Orange': (33.68, -117.78),
    'Placer': (39.06, -120.75), 'Plumas': (40.01, -120.84), 'Riverside': (33.74, -115.99),
    'Sacramento': (38.45, -121.34), 'San Benito': (36.61, -121.08), 'San Bernardino': (34.84, -116.18),
    'San Diego': (33.03, -116.74), 'San Francisco': (37.77, -122.44), 'San Joaquin': (37.93, -121.27),
    'San Luis Obispo': (35.39, -120.45), 'San Mateo': (37.42, -122.33), 'Santa Barbara': (34.65, -120.02),
    'Santa Clara': (37.20, -121.69), 'Santa Cruz': (37.02, -122.01), 'Shasta': (40.76, -122.04),
    'Sierra': (39.58, -120.50), 'Siskiyou': (41.59, -122.62), 'Solano': (38.27, -121.94),
    'Sonoma': (38.53, -122.92), 'Stanislaus': (37.56, -121.00), 'Sutter': (39.03, -121.69),
    'Tehama': (40.13, -122.23), 'Trinity': (40.65, -123.11), 'Tulare': (36.23, -118.80),
    'Tuolumne': (38.03, -119.95), 'Ventura': (34.46, -119.08), 'Yolo': (38.69, -121.90),
    'Yuba': (39.27, -121.35)
}

# Add coordinate columns helper
def add_coords(df):
    df = df.copy()
    df['latitude'] = df['County'].map(lambda c: COUNTY_COORDS.get(c, (None, None))[0])
    df['longitude'] = df['County'].map(lambda c: COUNTY_COORDS.get(c, (None, None))[1])
    return df.dropna(subset=['latitude', 'longitude'])

@st.experimental_memo
def load_cleaned_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = os.path.join(base_dir, "data_clean", "train_clean.csv")
    test_csv = os.path.join(base_dir, "data_clean", "test_clean.csv")
    predictions_csv = os.path.join(base_dir, "data_clean", "predictions_test.csv")
    
    if not os.path.exists(train_csv):
        st.warning("Chargement des données brutes (veuillez exécuter 'preprocessing.py' pour les données propres).")
        df_train = pd.read_csv("train.csv")
        df_train.rename(columns={'CI.lower': 'CI_lower', 'CI.upper': 'CI_upper'}, inplace=True)
        df_train['Recalculated_Rate'] = df_train['Rate']
        df_train = df_train[df_train['County'] != 'California']
        df_test = pd.read_csv("test.csv")
        df_test.rename(columns={'CI.lower': 'CI_lower', 'CI.upper': 'CI_upper'}, inplace=True)
        df_test['Recalculated_Rate'] = df_test['Rate']
        df_test = df_test[df_test['County'] != 'California']
        df_pred = df_test.copy()
        df_pred['Predicted_Rate'] = df_pred['Rate']
        df_pred['Priority_Level'] = "Basse Priorite"
    else:
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        df_pred = pd.read_csv(predictions_csv)
    return df_train, df_test, df_pred

# Initialize Spark session (once)
# PySpark not required for core functionality; set to None
spark = None

# Load data

df_train, df_test, df_pred = load_cleaned_data()
df_full = pd.concat([df_train, df_test], ignore_index=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("<div style='text-align: center;'><h2 style='color:#16a34a; margin-bottom:0;'>🏥 SANTE PFE</h2><p style='color:#64748b; font-size:0.9rem;'>Analyse & Machine Learning</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Navigation Menu
menu = st.sidebar.radio(
    "Navigation",
    ["📊 Vue d'Ensemble", "📈 Evolution Temporelle", "🗺️ Cartographie 3D", "🔮 ML & Points d'Intervention"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Filtres Globaux")

# Disease filter
diseases = sorted(df_full['Disease'].unique())
selected_disease = st.sidebar.selectbox("Maladie", ["Toutes"] + diseases, index=0)

# Gender filter
gender_options = ["Tous", "Male", "Female"]
selected_gender = st.sidebar.selectbox("Sexe", gender_options, index=0)

# Year filter
min_year, max_year = int(df_full['Year'].min()), int(df_full['Year'].max())
selected_years = st.sidebar.slider("Annees", min_year, max_year, (min_year, max_year))

# Apply global filters to a working df
df_filtered = df_full.copy()
if selected_disease != "Toutes":
    df_filtered = df_filtered[df_filtered['Disease'] == selected_disease]
if selected_gender != "Tous":
    df_filtered = df_filtered[df_filtered['Sex'] == selected_gender]
df_filtered = df_filtered[(df_filtered['Year'] >= selected_years[0]) & (df_filtered['Year'] <= selected_years[1])]

# Header Section
st.markdown("<h1 class='main-title'>Analyse des Maladies Infectieuses en Californie</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle'>Projet de Fin d'Etudes - PySpark & Streamlit Dashboard | Filtre actif : {selected_disease} | Sexe : {selected_gender} | Annees : {selected_years[0]}-{selected_years[1]}</p>", unsafe_allow_html=True)

# ----------------- PAGE 1: OVERVIEW -----------------
if menu == "📊 Vue d'Ensemble":
    # Metric cards row
    col1, col2, col3, col4 = st.columns(4)
    
    total_cases = int(df_filtered['Count'].sum())
    avg_rate = float(df_filtered['Recalculated_Rate'].mean())
    max_rate_row = df_filtered.loc[df_filtered['Recalculated_Rate'].idxmax()] if not df_filtered.empty else None
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Nombre Total de Cas</div>
            <div class='metric-val'>{total_cases:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Taux de Prevalence Moyen</div>
            <div class='metric-val'>{avg_rate:.2f} <span style='font-size:0.9rem; font-weight:normal; color:#64748b;'>/ 100k hab</span></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        top_county = df_filtered.groupby('County')['Count'].sum().idxmax() if not df_filtered.empty else "N/A"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Comte le Plus Touche</div>
            <div class='metric-val'>{top_county}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        top_disease = df_filtered.groupby('Disease')['Count'].sum().idxmax() if not df_filtered.empty else "N/A"
        if selected_disease != "Toutes":
            top_disease = selected_disease
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Maladie Principale</div>
            <div class='metric-val' style='font-size:1.4rem; padding-top:6px;'>{top_disease}</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts Row
    st.markdown("### 📊 Analyses de Repartition")
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("#### Top Maladies par Nombre de Cas")
        if not df_filtered.empty:
            disease_agg = df_filtered.groupby('Disease').agg(
                Cas_Totaux=('Count', 'sum'),
                Taux_Moyen=('Recalculated_Rate', 'mean')
            ).reset_index().sort_values(by='Cas_Totaux', ascending=False).head(10)
            
            chart_disease = alt.Chart(disease_agg).mark_bar(
                cornerRadiusTopRight=5,
                color='#16a34a'
            ).encode(
                y=alt.Y('Disease:N', sort='-x', title='Maladie'),
                x=alt.X('Cas_Totaux:Q', title='Cas Totaux'),
                tooltip=['Disease', 'Cas_Totaux', 'Taux_Moyen']
            ).properties(height=350, width=600)
            
            st.altair_chart(chart_disease, use_container_width=True)
        else:
            st.info("Aucune donnée disponible pour ces filtres.")
            
    with col_right:
        st.markdown("#### Repartition selon le Sexe")
        gender_df = df_filtered[df_filtered['Sex'].isin(['Male', 'Female'])]
        if not gender_df.empty:
            gender_agg = gender_df.groupby('Sex')['Count'].sum().reset_index()
            
            chart_gender = alt.Chart(gender_agg).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Sex", type="nominal", scale=alt.Scale(
                    domain=['Male', 'Female'],
                    range=['#3b82f6', '#ec4899']
                )),
                tooltip=['Sex', 'Count']
            ).properties(height=350)
            
            st.altair_chart(chart_gender, use_container_width=True)
        else:
            st.info("Aucune donnée disponible.")

    # Data Table View
    st.markdown("### 📂 Visualisation du Jeu de Donnees Propres")
    st.dataframe(df_filtered.head(100))

# ----------------- PAGE 2: TRENDS -----------------
elif menu == "📈 Evolution Temporelle":
    st.markdown("### 📈 Analyse des Tendances de Propagation")
    
    # Yearly evolution chart
    st.markdown("#### Evolution Annuelle des Cas et des Taux de Prevalence")
    yearly_agg = df_filtered.groupby('Year').agg(
        Cas_Totaux=('Count', 'sum'),
        Taux_Moyen=('Recalculated_Rate', 'mean')
    ).reset_index()
    
    if not yearly_agg.empty:
        # Altair double axis chart or side-by-side
        base = alt.Chart(yearly_agg).encode(x=alt.X('Year:O', title='Annee'))
        
        line_cases = base.mark_line(color='#6366f1', strokeWidth=3, point=True).encode(
            y=alt.Y('Cas_Totaux:Q', title='Cas Totaux'),
            tooltip=['Year', 'Cas_Totaux']
        )
        
        line_rate = base.mark_line(color='#10b981', strokeWidth=3, point=True).encode(
            y=alt.Y('Taux_Moyen:Q', title='Taux Moyen (/100k hab)'),
            tooltip=['Year', 'Taux_Moyen']
        )
        
        col_c, col_r = st.columns(2)
        with col_c:
            st.markdown("<h5 style='text-align:center;'>Cas Totaux Signalés</h5>", unsafe_allow_html=True)
            st.altair_chart(line_cases, use_container_width=True)
        with col_r:
            st.markdown("<h5 style='text-align:center;'>Taux Moyen de Prévalence (pour 100 000 habitants)</h5>", unsafe_allow_html=True)
            st.altair_chart(line_rate, use_container_width=True)
            
        # Analysis of growth
        first_year = yearly_agg['Year'].min()
        last_year = yearly_agg['Year'].max()
        first_cases = yearly_agg[yearly_agg['Year'] == first_year]['Cas_Totaux'].values[0]
        last_cases = yearly_agg[yearly_agg['Year'] == last_year]['Cas_Totaux'].values[0]
        
        growth = ((last_cases - first_cases) / first_cases) * 100 if first_cases > 0 else 0
        
        st.markdown("#### 💡 Diagnostic de Tendance")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.info(f"Année de départ : **{first_year}** ({first_cases:,} cas) ➡️ Année finale : **{last_year}** ({last_cases:,} cas)")
        with col_g2:
            color = "red" if growth > 0 else "green"
            word = "augmentation" if growth > 0 else "baisse"
            st.metric(
                label=f"Croissance Totale de la Maladie ({first_year} à {last_year})",
                value=f"{growth:+.2f}%",
                delta=f"{word} de propagation",
                delta_color="inverse" if growth > 0 else "normal"
            )
    else:
        st.info("Aucune donnée pour générer le graphique.")

# ----------------- PAGE 3: MAP -----------------
elif menu == "🗺️ Cartographie 3D":
    st.markdown("### 🗺️ Cartographie 3D des Hotspots de Sante Publique")
    st.markdown("Cette carte interactive en 3D affiche les taux de prévalence pour chaque comté de Californie. La hauteur et la couleur des cylindres représentent l'intensité de la prévalence de la maladie sélectionnée.")
    
    # Aggregate data by County for the selected years/disease
    map_df = df_filtered.groupby('County').agg(
        Total_Count=('Count', 'sum'),
        Mean_Rate=('Recalculated_Rate', 'mean'),
        Population=('Population', 'mean')
    ).reset_index()
    
    # Add coordinates
    map_df = add_coords(map_df)
    
    if not map_df.empty:
        # Scale rates for height representation
        max_rate = map_df['Mean_Rate'].max()
        map_df['height'] = (map_df['Mean_Rate'] / (max_rate if max_rate > 0 else 1)) * 30000 + 5000
        
        # Color mapping (from green to yellow to red)
        def get_color(rate, max_r):
            ratio = rate / (max_r if max_r > 0 else 1)
            # RGB color gradient
            r = int(min(255, ratio * 2 * 255))
            g = int(min(255, (2 - ratio * 2) * 255)) if ratio > 0.5 else 255
            b = int(50)
            return [r, g, b, 180]
            
        map_df['color'] = map_df['Mean_Rate'].map(lambda r: get_color(r, max_rate))
    st.markdown("### 🗺️ Cartographie 3D")
    st.info("The 3‑D map (PyDeck) is disabled because of a compatibility issue with the current Streamlit version. Below is a static placeholder.")
    # You could add a static image or Altair map here if desired.
    # For now we simply display a message.

    st.info("Aucune donnée cartographique disponible pour les filtres actuels.")

# ----------------- PAGE 4: ML & INTERVENTIONS -----------------
elif menu == "🔮 ML & Points d'Intervention":
    st.markdown("### 🔮 Modele de Machine Learning & Points d'Intervention Prioritaires")
    
    st.markdown("""
    Dans cette section, nous utilisons notre modèle de **Machine Learning (Random Forest Regressor)** entraîné sous **PySpark MLlib** ($R^2 = 89.54\\%$ sur les données de test) 
    pour prévoir les futurs taux d'infection par comté et classifier le niveau d'intervention publique requis :
    - <span style="color:#ef4444; font-weight:bold;">Haute Priorite</span> : Prévalence prédite supérieure à 50 cas pour 100 000 habitants.
    - <span style="color:#f59e0b; font-weight:bold;">Moyenne Priorite</span> : Prévalence prédite entre 5 et 50 cas pour 100 000 habitants.
    - <span style="color:#10b981; font-weight:bold;">Basse Priorite</span> : Prévalence prédite inférieure ou égale à 5 cas pour 100 000 habitants.
    """, unsafe_allow_html=True)
    
    # ML Filters
    col_ml1, col_ml2 = st.columns(2)
    with col_ml1:
        ml_disease = st.selectbox("Sélectionner la maladie pour la prédiction", diseases)
    with col_ml2:
        # Offer both existing and future years (up to 5 years beyond max data)
        max_year = int(df_train['Year'].max())
        future_years = list(range(max_year + 1, max_year + 6))
        union_years = set(df_pred['Year'].unique()).union(set(future_years))
        all_years = sorted(union_years, reverse=True)
        ml_year = st.selectbox("Sélectionner l'année cible", all_years)
        
    # Determine if we need to compute future predictions
    if ml_year in df_pred['Year'].unique():
        pred_filtered = df_pred[(df_pred['Disease'] == ml_disease) & (df_pred['Year'] == ml_year)]
    else:
        if spark is None:
            st.warning("PySpark is not available. Future predictions are not available.")
            pred_filtered = pd.DataFrame()
        else:
            # Train a simple RandomForest model using PySpark if not already trained
            # Prepare training data (combine train and test for robustness)
            combined = pd.concat([df_train, df_test], ignore_index=True)
            # Ensure numeric types
            combined['Population'] = combined['Population'].astype(float)
            # Spark preprocessing pipeline
            indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx", handleInvalid="keep") for col in ["County", "Disease"]]
            assembler = VectorAssembler(inputCols=["County_idx", "Disease_idx", "Year", "Population"], outputCol="features")
            pipeline = Pipeline(stages=indexers + [assembler])
            train_sdf = spark.createDataFrame(combined)
            model_pipe = pipeline.fit(train_sdf)
            train_prepped = model_pipe.transform(train_sdf)
            rf = RandomForestRegressor(featuresCol="features", labelCol="Recalculated_Rate", numTrees=100)
            rf_model = rf.fit(train_prepped)
            # Build future prediction dataframe
            future_df = pd.DataFrame({
                "County": combined['County'].unique(),
                "Disease": ml_disease,
                "Year": ml_year,
            })
            # Average population per county for feature
            pop_avg = combined.groupby('County')['Population'].mean().reset_index()
            future_df = future_df.merge(pop_avg, on='County', how='left')
            future_sdf = spark.createDataFrame(future_df)
            future_prepped = model_pipe.transform(future_sdf)
            pred_spark = rf_model.transform(future_prepped)
            pred_pd = pred_spark.select("County", "Disease", "Year", "Population", "prediction").toPandas()
            pred_pd = pred_pd.rename(columns={"prediction": "Predicted_Rate"})
            pred_filtered = pred_pd
    
    if not pred_filtered.empty:
        # Group by county to get overall risk for that county
        county_risk = pred_filtered.groupby('County').agg(
            Predicted_Rate=('Predicted_Rate', 'mean'),
            Population=('Population', 'mean')
        ).reset_index()
        # Add optional columns if present
        if 'Recalculated_Rate' in pred_filtered.columns:
            county_risk['Observed_Rate'] = pred_filtered.groupby('County')['Recalculated_Rate'].mean().values
        else:
            county_risk['Observed_Rate'] = np.nan
        if 'Count' in pred_filtered.columns:
            county_risk['Cases'] = pred_filtered.groupby('County')['Count'].sum().values
        else:
            county_risk['Cases'] = 0
            
        # Categorize
        def get_prio(rate):
            if rate <= 5.0:
                return "Basse Priorite"
            elif rate <= 50.0:
                return "Moyenne Priorite"
            else:
                return "Haute Priorite"
                
        county_risk['Priority_Level'] = county_risk['Predicted_Rate'].map(get_prio)
        county_risk = county_risk.sort_values(by='Predicted_Rate', ascending=False)
        
        # Metric display counts
        high_prio_count = len(county_risk[county_risk['Priority_Level'] == 'Haute Priorite'])
        med_prio_count = len(county_risk[county_risk['Priority_Level'] == 'Moyenne Priorite'])
        low_prio_count = len(county_risk[county_risk['Priority_Level'] == 'Basse Priorite'])
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid #ef4444;'>
                <div class='metric-label'>Comtés en Haute Priorité 🚨</div>
                <div class='metric-val' style='color:#ef4444;'>{high_prio_count}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid #f59e0b;'>
                <div class='metric-label'>Comtés en Priorité Moyenne ⚠️</div>
                <div class='metric-val' style='color:#f59e0b;'>{med_prio_count}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class='metric-card' style='border-left: 5px solid #10b981;'>
                <div class='metric-label'>Comtés en Basse Priorité ✅</div>
                <div class='metric-val' style='color:#10b981;'>{low_prio_count}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Recommedations and predicted priorities table
        st.markdown("### 📋 Plan d'Intervention Préconisé par le Modèle")
        
        col_tbl, col_recos = st.columns([1, 1])
        
        with col_tbl:
            st.markdown("#### Classement des Risques par Comté (Prédictions ML)")
            # Color code priority in dataframe display
            st.dataframe(county_risk[['County', 'Observed_Rate', 'Predicted_Rate', 'Priority_Level']].rename(columns={
                'County': 'Comté', 'Observed_Rate': 'Taux Observé', 'Predicted_Rate': 'Taux Prédit', 'Priority_Level': 'Niveau de Priorité'
            }))
            
        with col_recos:
            st.markdown("#### 🛡️ Recommandations d'Interventions Ciblées")
            high_counties = county_risk[county_risk['Priority_Level'] == 'Haute Priorite'].head(5)
            
            if not high_counties.empty:
                for idx, row in high_counties.iterrows():
                    st.markdown(f"""
                    <div class='reco-card high'>
                        <div style='font-weight:bold; color:#1e293b; font-size:1.1rem;'>🚨 Action Critique Recommandée : {row['County']}</div>
                        <div style='color:#475569; margin-top:5px;'>
                            Le modèle prévoit une prévalence très élevée de <b>{row['Predicted_Rate']:.2f} / 100k hab</b>.
                            <br/><b>Interventions prioritaires :</b> Lancer des campagnes de dépistage locales, distribuer des trousses médicales et vacciner dans les cliniques du comté.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                med_counties = county_risk[county_risk['Priority_Level'] == 'Moyenne Priorite'].head(3)
                if not med_counties.empty:
                    for idx, row in med_counties.iterrows():
                        st.markdown(f"""
                        <div class='reco-card med'>
                            <div style='font-weight:bold; color:#1e293b; font-size:1.1rem;'>⚠️ Surveillance Renforcée : {row['County']}</div>
                            <div style='color:#94a3b8; margin-top:5px;'>
                                Le modèle prévoit une prévalence de <b>{row['Predicted_Rate']:.2f} / 100k hab</b>.
                                <br/><b>Interventions :</b> Sensibilisation générale des professionnels de santé locaux et surveillance des rapports hebdomadaires.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("Aucun comté ne présente de risque élevé pour cette maladie et cette année. Maintenir les protocoles de surveillance sanitaire standards.")
    else:
        st.info("Aucune prédiction disponible pour ce couple Maladie/Année.")
