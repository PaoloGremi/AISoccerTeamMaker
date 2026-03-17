import streamlit as st
import pandas as pd
import numpy as np
import itertools
from sklearn.ensemble import RandomForestRegressor

st.title("⚽ Generatore di Squadre con AI Locale")

# ---------------------------------------
# 1. CARICA I CSV
# ---------------------------------------
players = pd.read_csv("players.csv")
stats = pd.read_csv("stats.csv")

# Il merge usa playerId -> id
df_full = stats.merge(players[["id", "name", "role"]], left_on="playerId", right_on="id", how="left")

# Usa il nome da players.csv se disponibile, altrimenti playerName da stats
df_full["displayName"] = df_full["name"].fillna(df_full["playerName"])

# Usa il ruolo da players.csv (più affidabile), fallback a quello di stats
df_full["role"] = df_full["role_x"].fillna(df_full["role_y"])

# ---------------------------------------
# 2. ALLENA MODELLO AI (Random Forest)
# Colonne reali da stats.csv:
# gamesPlayed, votesReceived, avgVote, bestVote, worstVote
# ---------------------------------------
features = ["avgVote", "bestVote", "worstVote", "gamesPlayed", "votesReceived"]

@st.cache_resource
def train_model(df):
    df_ml = df[df["gamesPlayed"] > 0].dropna(subset=features)
    X = df_ml[features]
    y = df_ml["avgVote"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df_full)

# ---------------------------------------
# 3. PREDIZIONE PUNTEGGIO GIOCATORI
# Chi non ha mai giocato riceve score basso di default
# ---------------------------------------
df_full = df_full.copy()
df_full["aiScore"] = model.predict(df_full[features].fillna(0))

# Penalizza chi non ha mai giocato
df_full.loc[df_full["gamesPlayed"] == 0, "aiScore"] = df_full["aiScore"].min()

# ---------------------------------------
# 4. SELEZIONE GIOCATORI
# ---------------------------------------
all_names = df_full["displayName"].dropna().sort_values().tolist()
available = st.multiselect("Giocatori disponibili", all_names)

if len(available) < 4:
    st.warning("Seleziona almeno 4 giocatori.")
    st.stop()

df_selected = df_full[df_full["displayName"].isin(available)].copy()
players_list = df_selected.to_dict("records")

# ---------------------------------------
# 5. ALGORITMO SQUADRE + RUOLI
# Ruoli attesi: P (portiere), D (difensore), C (centrocampista), A (attaccante)
# ---------------------------------------
def role_penalty(team_roles):
    penalty = 0
    # Ogni squadra dovrebbe avere almeno un difensore, centrocampista e attaccante
    for r in ["D", "C", "A"]:
        if r not in team_roles:
            penalty += 5
    return penalty

def evaluate_split(team1_idx, plist):
    team1_set = set(team1_idx)
    team1 = [plist[i] for i in team1_idx]
    team2 = [p for i, p in enumerate(plist) if i not in team1_set]

    sum1 = sum(p["aiScore"] for p in team1)
    sum2 = sum(p["aiScore"] for p in team2)

    roles1 = [p.get("role", "") for p in team1]
    roles2 = [p.get("role", "") for p in team2]

    penalty = role_penalty(roles1) + role_penalty(roles2)
    diff = abs(sum1 - sum2) + penalty
    return diff, team1, team2, sum1, sum2

# ---------------------------------------
# 6. GENERA SQUADRE AL CLICK
# ---------------------------------------
if st.button("⚽ Genera Squadre"):

    best = None
    n = len(players_list)
    half = n // 2

    with st.spinner("Calcolo la divisione ottimale..."):
        for comb in itertools.combinations(range(n), half):
            diff, t1, t2, s1, s2 = evaluate_split(comb, players_list)
            if best is None or diff < best["diff"]:
                best = {
                    "diff": diff,
                    "team1": t1,
                    "team2": t2,
                    "score1": s1,
                    "score2": s2
                }

    col1, col2 = st.columns(2)

    with col1:
        st.header("🔵 Squadra 1")
        for p in best["team1"]:
            games = p.get("gamesPlayed", 0)
            avg = p.get("avgVote", 0)
            st.write(f"• **{p['displayName']}** ({p.get('role', '?')}) — media {avg:.2f} in {games} partite")
        st.metric("Forza totale", round(best["score1"], 2))

    with col2:
        st.header("🔴 Squadra 2")
        for p in best["team2"]:
            games = p.get("gamesPlayed", 0)
            avg = p.get("avgVote", 0)
            st.write(f"• **{p['displayName']}** ({p.get('role', '?')}) — media {avg:.2f} in {games} partite")
        st.metric("Forza totale", round(best["score2"], 2))

    diff_display = round(best["diff"], 2)
    if diff_display < 1:
        st.success(f"✅ Squadre molto equilibrate! Differenza: {diff_display}")
    else:
        st.info(f"ℹ️ Differenza di forza: {diff_display}")
