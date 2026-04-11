# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:08:45 2026

@author: ADITYA
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import random
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")

    with zipfile.ZipFile("deliveries.zip", 'r') as zip_ref:
        zip_ref.extractall()

    deliveries = pd.read_csv("deliveries.csv")

    matches['date'] = pd.to_datetime(matches['date'])
    return matches, deliveries

matches, deliveries = load_data()

# =========================
# TRAIN MODEL
# =========================
@st.cache_data
def train_model(matches):
    df = matches.dropna(subset=['winner']).copy()

    df['team1_win'] = (df['winner'] == df['team1']).astype(int)

    X = pd.get_dummies(df[['team1','team2','venue']])
    y = df['team1_win']

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    return model, X.columns

model, cols = train_model(matches)

# =========================
# PREDICT MATCH
# =========================
def predict_match(model, cols, team1, team2, venue):
    input_df = pd.DataFrame([[team1, team2, venue]],
                            columns=['team1','team2','venue'])

    input_df = pd.get_dummies(input_df).reindex(columns=cols, fill_value=0)

    prob = model.predict_proba(input_df)[0]
    return prob  # [team2_prob, team1_prob]

# =========================
# SIMULATION
# =========================
def simulate_points(matches, model, cols, teams, forced=None):
    points = {team: 0 for team in teams}

    for _, row in matches.iterrows():
        t1, t2, venue = row['team1'], row['team2'], row['venue']

        if forced and (t1, t2) == forced[:2]:
            winner = forced[2]
        else:
            prob = predict_match(model, cols, t1, t2, venue)
            winner = t1 if random.random() < prob[1] else t2

        points[winner] += 2

    return points

def playoff_probability(matches, model, cols, teams, n=200, forced=None):
    qualify_count = {team: 0 for team in teams}

    for _ in range(n):
        pts = simulate_points(matches, model, cols, teams, forced)

        top4 = sorted(pts, key=pts.get, reverse=True)[:4]

        for t in top4:
            qualify_count[t] += 1

    return {t: round((qualify_count[t]/n)*100,2) for t in teams}

# =========================
# UI
# =========================
st.title("🏏 IPL Predictive Analytics System")
st.markdown("### 🚀 ML + Simulation + Probability Dashboard")
st.markdown("---")

teams = sorted(matches['team1'].unique())

# =========================
# MATCH PREDICTION
# =========================
st.subheader("🤖 Match Prediction")

team1 = st.selectbox("Team 1", teams)
team2 = st.selectbox("Team 2", teams)

venue_pred = st.selectbox("Venue", matches['venue'].dropna().unique())

if st.button("Predict Match"):
    prob = predict_match(model, cols, team1, team2, venue_pred)

    st.success(f"{team1} Win Probability: {round(prob[1]*100,2)}%")
    st.info(f"{team2} Win Probability: {round(prob[0]*100,2)}%")

# =========================
# SCENARIO SIMULATOR
# =========================
st.subheader("🔮 Scenario Simulator")

sim_team1 = st.selectbox("Scenario Team 1", teams, key="sim1")
sim_team2 = st.selectbox("Scenario Team 2", teams, key="sim2")

forced_winner = st.selectbox("Force Winner", [sim_team1, sim_team2])

forced_match = (sim_team1, sim_team2, forced_winner)

# =========================
# PLAYOFF PROBABILITY
# =========================
st.subheader("🏆 Playoff Probability")

if st.button("Run Simulation"):
    probs = playoff_probability(matches, model, cols, teams, n=200, forced=forced_match)

    df_probs = pd.DataFrame({
        "Team": list(probs.keys()),
        "Playoff %": list(probs.values())
    }).sort_values(by="Playoff %", ascending=False)

    st.dataframe(df_probs, use_container_width=True)

    # Graph
    st.subheader("📊 Playoff Probability Chart")

    fig, ax = plt.subplots()
    ax.bar(df_probs['Team'], df_probs['Playoff %'])
    plt.xticks(rotation=45)
    st.pyplot(fig)

# =========================
# PLAYER SEARCH
# =========================
st.subheader("🔍 Player Search")

player = st.text_input("Enter Player Name")

if player:
    player = player.lower()

    runs = deliveries[deliveries['batter'].str.lower() == player]['batsman_runs'].sum()
    wickets = deliveries[deliveries['bowler'].str.lower() == player]['is_wicket'].sum()

    if runs == 0 and wickets == 0:
        st.warning("Player not found ❗")
    else:
        st.write(f"🏏 Runs: {runs}")
        st.write(f"🎯 Wickets: {wickets}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("🚀 Built by Aditya Herwade | IPL ML Simulation System")