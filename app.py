import streamlit as st
import pandas as pd
import numpy as np
import itertools
import re
import warnings
warnings.filterwarnings("ignore", category=Warning, module="urllib3")
from sklearn.ensemble import RandomForestRegressor

st.title("⚽ Generatore di Squadre con AI Locale")

# ---------------------------------------
# 1. CARICA I CSV
# ---------------------------------------
players = pd.read_csv("players.csv")
stats   = pd.read_csv("stats.csv")
vc      = pd.read_csv("votes_comments.csv")

# Merge stats + players
df_full = stats.merge(players[["id", "name", "role"]], left_on="playerId", right_on="id", how="left")
df_full["displayName"] = df_full["name"].fillna(df_full["playerName"])
df_full["role"]        = df_full["role_x"].fillna(df_full["role_y"])

# ---------------------------------------
# 2. SENTIMENT DAI COMMENTI (dizionario italiano)
# ---------------------------------------
POSITIVE_WORDS = {
    "ottima","ottimo","grande","bello","bella","decisivo","decisiva","eccellente",
    "bravo","brava","fantastico","fantastica","miglioramento","qualità","assist",
    "goal","gol","doppietta","tripletta","indistruttibile","costante","crescita",
    "preciso","precisa","leader","determinante","concreto","concreta","intelligente",
    "tecnico","tecnica","dominante","protagonista","esplosivo","esplosiva"
}
NEGATIVE_WORDS = {
    "male","peggio","errore","sbagliato","sbagliata","perde","perso","colpa",
    "timido","lento","lenta","svarione","regalo","divora","pecca","ritardo",
    "impreciso","imprecisa","affaticato","affaticata","assenza","ingenuo","ingenua",
    "distratto","distratta","nervoso","nervosa"
}

def sentiment_score(text):
    if not isinstance(text, str):
        return 0.0
    words = re.findall(r'\w+', text.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0

vc["sentiment"] = vc["comment"].apply(sentiment_score)

# Aggregazione per giocatore da votes_comments
vc_agg = vc.groupby("playerId").agg(
    avgVoteVC    = ("vote",      "mean"),
    sentimentAvg = ("sentiment", "mean"),
    commentCount = ("vote",      "count"),
    voteStd      = ("vote",      "std")
).reset_index()
vc_agg["voteStd"] = vc_agg["voteStd"].fillna(0)

# Merge nel df principale
df_full = df_full.merge(vc_agg, on="playerId", how="left")
df_full["avgVoteVC"]    = df_full["avgVoteVC"].fillna(df_full["avgVote"])
df_full["sentimentAvg"] = df_full["sentimentAvg"].fillna(0)
df_full["commentCount"] = df_full["commentCount"].fillna(0)
df_full["voteStd"]      = df_full["voteStd"].fillna(0)

# ---------------------------------------
# 3. ALLENA MODELLO AI (Random Forest)
# Feature arricchite con dati da votes_comments
# ---------------------------------------
features = [
    "avgVote", "bestVote", "worstVote", "gamesPlayed", "votesReceived",
    "avgVoteVC", "sentimentAvg", "commentCount", "voteStd"
]

@st.cache_resource
def train_model(df):
    df_ml = df[df["gamesPlayed"] > 0].dropna(subset=features)
    X = df_ml[features]
    y = df_ml["avgVoteVC"]   # target: media voti da votes_comments (più granulare)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df_full)

# ---------------------------------------
# 4. PREDIZIONE PUNTEGGIO GIOCATORI
# ---------------------------------------
df_full = df_full.copy()
df_full["aiScore"] = model.predict(df_full[features].fillna(0))
df_full.loc[df_full["gamesPlayed"] == 0, "aiScore"] = df_full["aiScore"].min()

# ---------------------------------------
# 5. SELEZIONE GIOCATORI
# ---------------------------------------
all_names = df_full["displayName"].dropna().sort_values().tolist()
available = st.multiselect("Giocatori disponibili", all_names)

if len(available) < 4:
    st.warning("Seleziona almeno 4 giocatori.")
    st.stop()

df_selected  = df_full[df_full["displayName"].isin(available)].copy()
players_list = df_selected.to_dict("records")

# Ultimi 3 commenti per giocatore (per il prompt Ollama)
last_comments = (
    vc.sort_values("matchDate")
      .groupby("playerId")
      .tail(3)
      .groupby("playerId")
      .apply(lambda g: [
          {"date": row["matchDate"], "vote": row["vote"], "comment": row["comment"]}
          for _, row in g.iterrows()
      ])
      .to_dict()
)

# ---------------------------------------
# 6. ALGORITMO SQUADRE + RUOLI
# ---------------------------------------
def role_penalty(team_roles):
    penalty = 0
    for r in ["D", "C", "A"]:
        if r not in team_roles:
            penalty += 5
    return penalty

def evaluate_split(team1_idx, plist):
    team1_set = set(team1_idx)
    team1 = [plist[i] for i in team1_idx]
    team2 = [p for i, p in enumerate(plist) if i not in team1_set]
    sum1  = sum(p["aiScore"] for p in team1)
    sum2  = sum(p["aiScore"] for p in team2)
    roles1 = [p.get("role", "") for p in team1]
    roles2 = [p.get("role", "") for p in team2]
    penalty = role_penalty(roles1) + role_penalty(roles2)
    diff = abs(sum1 - sum2) + penalty
    return diff, team1, team2, sum1, sum2

# ---------------------------------------
# 7. GENERA SQUADRE AL CLICK
# ---------------------------------------
if st.button("⚽ Genera Squadre"):

    best = None
    n    = len(players_list)
    half = n // 2

    with st.spinner("Calcolo la divisione ottimale..."):
        for comb in itertools.combinations(range(n), half):
            diff, t1, t2, s1, s2 = evaluate_split(comb, players_list)
            if best is None or diff < best["diff"]:
                best = {"diff": diff, "team1": t1, "team2": t2, "score1": s1, "score2": s2}

    col1, col2 = st.columns(2)

    with col1:
        st.header("🏳️ Squadra BIANCA")
        for p in best["team1"]:
            sentiment_label = ""
            if p.get("sentimentAvg", 0) > 0.3:
                sentiment_label = " 🔥"
            elif p.get("sentimentAvg", 0) < -0.1:
                sentiment_label = " 📉"
            st.write(
                f"• **{p['displayName']}** ({p.get('role','?')}) — "
                f"media {p.get('avgVoteVC', p.get('avgVote',0)):.2f} "
                f"su {int(p.get('commentCount',0))} partite{sentiment_label}"
            )
        st.metric("Forza totale", round(best["score1"], 2))

    with col2:
        st.header("🏳️‍🌈 Squadra COLORATA")
        for p in best["team2"]:
            sentiment_label = ""
            if p.get("sentimentAvg", 0) > 0.3:
                sentiment_label = " 🔥"
            elif p.get("sentimentAvg", 0) < -0.1:
                sentiment_label = " 📉"
            st.write(
                f"• **{p['displayName']}** ({p.get('role','?')}) — "
                f"media {p.get('avgVoteVC', p.get('avgVote',0)):.2f} "
                f"su {int(p.get('commentCount',0))} partite{sentiment_label}"
            )
        st.metric("Forza totale", round(best["score2"], 2))

    diff_display = round(best["diff"], 2)
    if diff_display < 1:
        st.success(f"✅ Squadre molto equilibrate! Differenza: {diff_display}")
    else:
        st.info(f"ℹ️ Differenza di forza: {diff_display}")

    # ---------------------------------------
    # 8. DESCRIZIONE VERBALE VIA OLLAMA
    # ---------------------------------------
    def format_team_for_prompt(team, score):
        lines = []
        for p in team:
            pid = p.get("playerId", "")
            comments_text = ""
            if pid in last_comments:
                snippets = [
                    f'[{c["date"]}, voto {c["vote"]}] {c["comment"]}'
                    for c in last_comments[pid]
                ]
                comments_text = "\n  Ultimi commenti:\n  " + "\n  ".join(snippets)
            lines.append(
                f"- {p['displayName']} ({p.get('role','?')}): "
                f"media {p.get('avgVoteVC', p.get('avgVote',0)):.2f}, "
                f"sentiment {'positivo' if p.get('sentimentAvg',0) > 0 else 'negativo' if p.get('sentimentAvg',0) < 0 else 'neutro'}"
                f"{comments_text}"
            )
        lines.append(f"Forza totale stimata: {score:.2f}")
        return "\n".join(lines)

    prompt = f"""Sei un commentatore sportivo di calcetto amatoriale.
Ti vengono fornite due squadre generate automaticamente da un algoritmo di bilanciamento.
Per ogni giocatore hai la media voti, il sentiment dei commenti storici e gli ultimi commenti reali delle partite.
Scrivi una descrizione in italiano, tono informale e divertente (4-6 frasi), che:
- Spieghi perché le squadre sono bilanciate
- Citi almeno un dettaglio concreto dai commenti storici per rendere il testo vivace
- Evidenzi i giocatori di punta di ciascuna squadra
Non inventare statistiche non presenti nei dati.

SQUADRA BIANCA (🏳️):
{format_team_for_prompt(best['team1'], best['score1'])}

SQUADRA COLORATA (🏳️‍🌈):
{format_team_for_prompt(best['team2'], best['score2'])}

Differenza di forza tra le squadre: {diff_display}
"""

    st.divider()
    st.subheader("🎙️ Commento AI")

    with st.spinner("Ollama sta scrivendo il commento..."):
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": False},
                timeout=60
            )
            response.raise_for_status()
            narrative = response.json().get("response", "").strip()
            st.markdown(narrative)
        except requests.exceptions.ConnectionError:
            st.error("❌ Ollama non raggiungibile. Assicurati che sia avviato con `ollama serve`.")
        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout: Ollama ha impiegato troppo. Riprova o usa un modello più leggero.")
        except Exception as e:
            st.error(f"❌ Errore durante la chiamata a Ollama: {e}")
