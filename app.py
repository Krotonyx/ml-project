import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Travel City Recommender", page_icon="🌍", layout="wide")


# --- CORE LOGIC (CACHED) ---
@st.cache_resource
def load_data_and_train():
    # Load dataset
    df = pd.read_csv('Worldwide Travel Cities Dataset (Ratings and Climate).csv')

    # Preprocessing (From your original code)
    df_clean = df.drop(columns=['id', 'latitude', 'longitude', 'short_description'])
    df_clean['avg_temp_monthly'] = df_clean['avg_temp_monthly'].apply(json.loads)
    df_clean['ideal_durations'] = df_clean['ideal_durations'].apply(json.loads)

    # Encoding
    oe = OrdinalEncoder(categories=[['Budget', 'Mid-range', 'Luxury']])
    df_clean['budget_level'] = oe.fit_transform(df_clean[['budget_level']]).astype(int)

    # Region Encoding
    regions = ['europe', 'asia', 'africa', 'oceania', 'middle_east', 'north_america', 'south_america']
    for r in regions:
        df_clean[r] = (df_clean['region'] == r).astype(int)

    # Model Setup
    feature_cols = ['culture', 'adventure', 'nature', 'beaches', 'nightlife',
                    'cuisine', 'wellness', 'urban', 'seclusion'] + regions
    X = df_clean[feature_cols]
    y = df_clean['budget_level']

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return df_clean, model, feature_cols, regions


df, model_gbc, features, region_list = load_data_and_train()

# --- HELPER FUNCTIONS ---
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


def build_vector(trip_types):
    base_features = ['culture', 'adventure', 'nature', 'beaches', 'nightlife', 'cuisine', 'wellness', 'urban',
                     'seclusion']
    combined = {col: 0 for col in base_features}
    for t in trip_types:
        for col in base_features:
            combined[col] += TRIP_TYPE_VECTORS[t][col]
    return {col: round(combined[col] / len(trip_types)) for col in base_features}


# --- SIDEBAR UI ---
st.sidebar.header("🗺️ Trip Planner")
selected_types = st.sidebar.multiselect("Select Trip Types", list(TRIP_TYPE_VECTORS.keys()), default=['culture'])
selected_region = st.sidebar.selectbox("Preferred Region", region_list)
selected_month = st.sidebar.slider("Travel Month", 1, 12, 6)
selected_dur = st.sidebar.selectbox("Duration", ["Weekend", "Short trip", "One week", "Long trip"])

# --- MAIN DASHBOARD ---
st.title("Travel City Recommender")
st.markdown("Predicting your ideal destination using **Gradient Boosting** and **K-Nearest Neighbors**.")

if st.sidebar.button("Generate Recommendations") and selected_types:
    # 1. Prediction Logic
    prefs = build_vector(selected_types)
    pref_values = [prefs[col] for col in
                   ['culture', 'adventure', 'nature', 'beaches', 'nightlife', 'cuisine', 'wellness', 'urban',
                    'seclusion']]
    region_values = [1 if r == selected_region else 0 for r in region_list]
    full_input = pd.DataFrame([pref_values + region_values], columns=features)

    pred_budget_idx = model_gbc.predict(full_input)[0]
    budget_labels = {0: 'Budget', 1: 'Mid-range', 2: 'Luxury'}
    predicted_budget = budget_labels[pred_budget_idx]

    # 2. Filtering
    filtered = df[(df['region'] == selected_region) &
                  (df['budget_level'] == pred_budget_idx) &
                  (df['ideal_durations'].apply(lambda x: selected_dur in x)) &
                  (df['avg_temp_monthly'].apply(lambda x: x[str(selected_month)]['avg'] >= 15))]

    if filtered.empty:
        st.error("No cities match your strict weather/duration filters. Try a different month or duration.")
    else:
        # 3. Matching
        knn = NearestNeighbors(n_neighbors=min(5, len(filtered)), metric='cosine')
        knn.fit(filtered[['culture', 'adventure', 'nature', 'beaches', 'nightlife', 'cuisine', 'wellness', 'urban',
                          'seclusion']])
        dist, idx = knn.kneighbors([pref_values])

        results = filtered.iloc[idx[0]].copy()
        results['Match Score'] = (1 - dist[0]).round(3)
        results['budget_level'] = results['budget_level'].map(budget_labels)

        # 4. Display Results
        st.subheader(f"✨ Predicted Budget: {predicted_budget}")
        st.write("Top 5 city matches in the selected region:")
        st.table(results[['city', 'country', 'region', 'budget_level', 'Match Score']])
else:
    st.info("Please select trip types in the sidebar and click 'Generate Recommendations'.")
