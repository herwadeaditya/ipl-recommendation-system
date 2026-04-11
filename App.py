# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:08:45 2026

@author: ADITYA
"""


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import zipfile

# LOAD DATA
@st.cache_data
def load_data():
    matches = pd.read_csv("matches.csv")

    # unzip deliveries
    with zipfile.ZipFile("deliveries.zip", 'r') as zip_ref:
        zip_ref.extractall()

    deliveries = pd.read_csv("deliveries.csv")

    matches['date'] = pd.to_datetime(matches['date'])
    return matches, deliveries

matches, deliveries = load_data()

# TITLE 
st.title("🏏 IPL Smart Analytics Dashboard")
st.markdown("### 📊 Advanced IPL Recommendation System")
st.markdown("---")

# INPUT
team = st.selectbox("Select Team", sorted(matches['team1'].unique()))
year = st.selectbox("Select Season", sorted(matches['season'].unique()))

st.write(f"Showing results for **{team}** in **{year}**")

# FILTER
team_matches = matches[
    ((matches['team1'] == team) | (matches['team2'] == team)) &
    (matches['season'] == year)
].copy()

# KPI CARDS
wins = team_matches[team_matches['winner'] == team].shape[0]
total = team_matches.shape[0]

st.subheader("📊 Team Stats")

col1, col2, col3 = st.columns(3)

col1.metric("Matches", total)
col2.metric("Wins", wins)
col3.metric("Win Rate", f"{round((wins/total)*100,2)}%" if total > 0 else "0%")

# RECENT MATCHES
st.subheader("🔥 Recent Matches")

recent = team_matches.sort_values(by='date', ascending=False).head(5).reset_index(drop=True)
recent.insert(0, 'No', range(1, len(recent)+1))

st.dataframe(
    recent[['No','date','team1','team2','winner']],
    use_container_width=True,
    hide_index=True,
    column_config={
        "No": st.column_config.NumberColumn(width="small")
    }
)

# SCORING
def match_score(row):
    score = 0

    if "Mumbai Indians" in [row['team1'], row['team2']]:
        score += 5

    if row['result_margin'] <= 5:
        score += 5
    elif row['result_margin'] <= 10:
        score += 3

    if row['winner'] == team:
        score += 2

    return score

def explain(row):
    reasons = []

    if "Mumbai Indians" in [row['team1'], row['team2']]:
        reasons.append("Rivalry")
    if row['result_margin'] <= 10:
        reasons.append("Close")
    if row['winner'] == team:
        reasons.append("Win")

    return ", ".join(reasons)

team_matches['score'] = team_matches.apply(match_score, axis=1)
team_matches['reason'] = team_matches.apply(explain, axis=1)

recommended = team_matches.sort_values(by='score', ascending=False)
recommended = recommended[recommended['score'] > 0].head(5).reset_index(drop=True)
recommended.insert(0, 'No', range(1, len(recommended)+1))

st.subheader("⭐ Recommended Matches")

st.dataframe(
    recommended[['No','date','team1','team2','winner','score','reason']],
    use_container_width=True,
    hide_index=True,
    column_config={
        "No": st.column_config.NumberColumn(width="small")
    }
)

# HEAD TO HEAD 
st.subheader("⚔️ Head-to-Head")

opponent = st.selectbox(
    "Select Opponent",
    [t for t in sorted(matches['team1'].unique()) if t != team]
)

h2h = matches[
    ((matches['team1'] == team) & (matches['team2'] == opponent)) |
    ((matches['team2'] == team) & (matches['team1'] == opponent))
]

wins_team = h2h[h2h['winner'] == team].shape[0]
wins_opp = h2h[h2h['winner'] == opponent].shape[0]

st.write(f"{team} Wins: {wins_team}")
st.write(f"{opponent} Wins: {wins_opp}")

# TOP BATSMEN
team_batting = deliveries[deliveries['batting_team'] == team]

top_batsmen = team_batting.groupby('batter')['batsman_runs'] \
    .sum().sort_values(ascending=False).head(5).reset_index()

top_batsmen.columns = ['Player', 'Runs']
top_batsmen.insert(0, 'Rank', range(1, len(top_batsmen)+1))

st.subheader("🏏 Top Batsmen")

st.dataframe(
    top_batsmen,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Player": st.column_config.TextColumn(width="large"),
        "Runs": st.column_config.NumberColumn(width="medium"),
    }
)

# GRAPH
st.subheader("📈 Batsmen Performance")

fig, ax = plt.subplots()
ax.bar(top_batsmen['Player'], top_batsmen['Runs'])
plt.xticks(rotation=30)
st.pyplot(fig)

# TOP BOWLERS
team_bowling = deliveries[deliveries['bowling_team'] == team]
team_bowling = team_bowling[team_bowling['is_wicket'] == 1]

top_bowlers = team_bowling.groupby('bowler')['is_wicket'] \
    .sum().sort_values(ascending=False).head(5).reset_index()

top_bowlers.columns = ['Player', 'Wickets']
top_bowlers.insert(0, 'Rank', range(1, len(top_bowlers)+1))

st.subheader("🎯 Top Bowlers")

st.dataframe(
    top_bowlers,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Player": st.column_config.TextColumn(width="large"),
        "Wickets": st.column_config.NumberColumn(width="medium"),
    }
)

# PLAYER SEARCH
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

# TOSS ANALYSIS 
st.subheader("🪙 Toss Impact")

toss_win = matches[matches['toss_winner'] == matches['winner']].shape[0]
total_matches = matches.shape[0]

st.write(f"Toss win leads to match win: {round((toss_win/total_matches)*100,2)}%")

# VENUE ANALYSIS
st.subheader("🏟️ Venue Analysis")

venue = st.selectbox("Select Venue", matches['venue'].dropna().unique())

venue_stats = matches[matches['venue'] == venue]['winner'].value_counts().head(5)
st.write(venue_stats)

# FOOTER
st.markdown("---")
st.markdown("🚀 Built by Aditya Herwade | IPL Advanced Analytics System")