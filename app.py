import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors

# --- PAGE CONFIG ---
st.set_page_config(page_title="GML Travel Recommender", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.65), rgba(15, 23, 42, 0.65)), 
                    url('https://images.unsplash.com/photo-1499856871958-5b9627545d1a?q=80&w=2000');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 30px;
        color: white;
        margin-bottom: 20px;
    }
    .header-text {
        text-align: center;
        margin-bottom: 25px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        padding-bottom: 15px;
    }
    label, p, h1, h2, h3 { color: white !important; }
    .stButton>button {
        background: #0ea5e9;
        color: white;
        border-radius: 12px;
        width: 100%;
        padding: 12px;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #0284c7;
        transform: translateY(-2px);
    }
    /* Style for the checkbox grid */
    .checkbox-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# --- DATA & MODEL CACHING ---
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('Worldwide Travel Cities Dataset (Ratings and Climate).csv')
    df_clean = df.drop(columns=['id', 'latitude', 'longitude', 'short_description'])
    df_clean['avg_temp_monthly'] = df_clean['avg_temp_monthly'].apply(json.loads)
    df_clean['ideal_durations'] = df_clean['ideal_durations'].apply(json.loads)
    oe = OrdinalEncoder(categories=[['Budget', 'Mid-range', 'Luxury']])
    df_clean['budget_level_idx'] = oe.fit_transform(df_clean[['budget_level']]).astype(int)
    df_encoded = pd.get_dummies(df_clean['region']).astype(int)
    df_final = pd.concat([df_clean, df_encoded], axis=1)
    return df_final


@st.cache_resource
def train_models(df):
    regions_list = ['europe', 'asia', 'africa', 'oceania', 'middle_east', 'north_america', 'south_america']
    features = ['culture', 'adventure', 'nature', 'beaches', 'nightlife',
                'cuisine', 'wellness', 'urban', 'seclusion'] + regions_list
    X = df[features]
    y = df['budget_level_idx']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model, features


# Initialize
df = load_and_clean_data()
model_gbc, FEATURE_COLS_FULL = train_models(df)

# --- MAPPINGS ---
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

# FIX 1: Display names for Regions
region_map = {
    "Europe": "europe",
    "Asia": "asia",
    "Africa": "africa",
    "Oceania": "oceania",
    "Middle East": "middle_east",
    "North America": "north_america",
    "South America": "south_america"
}

TRIP_TYPE_VECTORS = {
    'beach': {'culture': 2, 'adventure': 2, 'nature': 3, 'beaches': 5, 'nightlife': 2, 'cuisine': 2, 'wellness': 3,
              'urban': 1, 'seclusion': 3},
    'adventure': {'culture': 2, 'adventure': 5, 'nature': 4, 'beaches': 2, 'nightlife': 1, 'cuisine': 2, 'wellness': 2,
                  'urban': 1, 'seclusion': 3},
    'culture': {'culture': 5, 'adventure': 2, 'nature': 2, 'beaches': 1, 'nightlife': 3, 'cuisine': 4, 'wellness': 2,
                'urban': 4, 'seclusion': 1},
    'food': {'culture': 4, 'adventure': 1, 'nature': 2, 'beaches': 2, 'nightlife': 3, 'cuisine': 5, 'wellness': 2,
             'urban': 4, 'seclusion': 1},
    'nightlife': {'culture': 2, 'adventure': 2, 'nature': 1, 'beaches': 3, 'nightlife': 5, 'cuisine': 3, 'wellness': 1,
                  'urban': 5, 'seclusion': 1},
    'wellness': {'culture': 2, 'adventure': 2, 'nature': 4, 'beaches': 3, 'nightlife': 1, 'cuisine': 3, 'wellness': 5,
                 'urban': 1, 'seclusion': 4},
    'nature': {'culture': 2, 'adventure': 4, 'nature': 5, 'beaches': 2, 'nightlife': 1, 'cuisine': 2, 'wellness': 3,
               'urban': 1, 'seclusion': 4},
    'urban': {'culture': 4, 'adventure': 2, 'nature': 1, 'beaches': 1, 'nightlife': 4, 'cuisine': 4, 'wellness': 2,
              'urban': 5, 'seclusion': 1},
}

# --- UI LAYOUT ---
st.markdown(
    '<div class="header-text"><h1>Global Wanderer</h1><p>Discover your perfect destination using Machine Learning.</p></div>',
    unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # Region with improved display names
        selected_region_display = st.selectbox("Preferred Region", list(region_map.keys()))
        region_input = region_map[selected_region_display]

        selected_month_name = st.selectbox("Travel Month", list(month_map.keys()), index=5)
        month_input = month_map[selected_month_name]

    with col2:
        duration_input = st.selectbox("Trip Duration", ["Weekend", "Short trip", "One week", "Long trip"])

        # FIX 2: Checkboxes instead of Dropdown for Interests
        st.write("Travel Interests")
        interest_options = ['beach', 'adventure', 'culture', 'food', 'nightlife', 'wellness', 'nature', 'urban']

        # Arrange checkboxes in a mini-grid using columns
        c1, c2 = st.columns(2)
        trip_types = []
        for i, option in enumerate(interest_options):
            with (c1 if i % 2 == 0 else c2):
                if st.checkbox(option.capitalize(), key=option):
                    trip_types.append(option)

    if st.button("Generate Recommendations"):
        if not trip_types:
            st.warning("Please select at least one travel interest.")
        else:
            # Prediction Logic
            base_features = ['culture', 'adventure', 'nature', 'beaches', 'nightlife', 'cuisine', 'wellness', 'urban',
                             'seclusion']
            prefs = {col: 0 for col in base_features}
            for t in trip_types:
                for col in base_features:
                    prefs[col] += TRIP_TYPE_VECTORS[t][col]
            avg_prefs = [round(prefs[col] / len(trip_types)) for col in base_features]

            region_cols = ['europe', 'asia', 'africa', 'oceania', 'middle_east', 'north_america', 'south_america']
            region_values = [1 if col == region_input else 0 for col in region_cols]
            full_input = avg_prefs + region_values

            pred_budget_idx = model_gbc.predict([full_input])[0]
            budget_labels = {0: 'Budget', 1: 'Mid-range', 2: 'Luxury'}

            st.success(f"Our Model predicts a **{budget_labels[pred_budget_idx]}** budget level for this journey.")

            # Filtering and KNN
            filtered = df[(df['region'] == region_input) &
                          (df['budget_level_idx'] == pred_budget_idx) &
                          (df['ideal_durations'].apply(lambda x: duration_input in x)) &
                          (df['avg_temp_monthly'].apply(lambda x: x[str(month_input)]['avg'] >= 15))]

            if filtered.empty:
                st.info("No exact matches found. Try changing the travel month or trip duration!")
            else:
                knn = NearestNeighbors(n_neighbors=min(5, len(filtered)), metric='cosine')
                knn.fit(filtered[base_features])
                dist, indices = knn.kneighbors([avg_prefs])

                results = filtered.iloc[indices[0]].copy()
                results['Match Score'] = (1 - dist[0]).round(3)

                st.markdown("### Top Destination Matches")
                final_display = results[['city', 'country', 'budget_level', 'Match Score']].rename(columns=str.title)
                st.dataframe(final_display, use_container_width=True, hide_index=True)

    st.markdown('</div>', unsafe_allow_html=True)
