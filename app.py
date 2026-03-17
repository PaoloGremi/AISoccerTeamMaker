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
matches = pd.read_csv("matches.csv")

# Merge stats + players
df_full = stats.merge(players[["id", "name", "role"]], left_on="playerId", right_on="id", how="left")
df_full["displayName"] = df_full["name"].fillna(df_full["playerName"])
df_full["role"]        = df_full["role_x"].fillna(df_full["role_y"])

# ---------------------------------------
# 2. FEATURE DA MATCHES (vittorie, MVP, hustle)
# ---------------------------------------
records = []
for _, row in matches.iterrows():
    teamA = row["teamA"].split("|")
    teamB = row["teamB"].split("|")
    sA, sB = row["scoreA"], row["scoreB"]
    if sA > sB:    resA, resB = "W", "L"
    elif sA < sB:  resA, resB = "L", "W"
    else:           resA, resB = "D", "D"
    for pid in teamA:
        records.append({"playerId": pid, "result": resA})
    for pid in teamB:
        records.append({"playerId": pid, "result": resB})

df_res = pd.DataFrame(records)
match_agg = df_res.groupby("playerId").agg(
    wins   = ("result", lambda x: (x == "W").sum()),
    draws  = ("result", lambda x: (x == "D").sum()),
    losses = ("result", lambda x: (x == "L").sum()),
).reset_index()

mvp_counts      = matches["mvp"].value_counts().rename("mvpCount")
hustle_counts   = matches["hustlePlayer"].value_counts().rename("hustleCount")
bestgoal_counts = matches["bestGoalPlayer"].value_counts().rename("bestGoalCount")

match_agg = match_agg.set_index("playerId")
match_agg = match_agg.join(mvp_counts).join(hustle_counts).join(bestgoal_counts).fillna(0).reset_index()
match_agg["winRate"] = (
    match_agg["wins"] / (match_agg["wins"] + match_agg["draws"] + match_agg["losses"])
).fillna(0)

df_full = df_full.merge(match_agg, on="playerId", how="left")
for col in ["wins", "draws", "losses", "mvpCount", "hustleCount", "bestGoalCount", "winRate"]:
    df_full[col] = df_full[col].fillna(0)

# ---------------------------------------
# 3. SENTIMENT DAI COMMENTI
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
    pos   = sum(1 for w in words if w in POSITIVE_WORDS)
    neg   = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    return (pos - neg) / total if total > 0 else 0.0

vc["sentiment"] = vc["comment"].apply(sentiment_score)

vc_agg = vc.groupby("playerId").agg(
    avgVoteVC    = ("vote",      "mean"),
    sentimentAvg = ("sentiment", "mean"),
    commentCount = ("vote",      "count"),
    voteStd      = ("vote",      "std")
).reset_index()
vc_agg["voteStd"] = vc_agg["voteStd"].fillna(0)

df_full = df_full.merge(vc_agg, on="playerId", how="left")
df_full["avgVoteVC"]    = df_full["avgVoteVC"].fillna(df_full["avgVote"])
df_full["sentimentAvg"] = df_full["sentimentAvg"].fillna(0)
df_full["commentCount"] = df_full["commentCount"].fillna(0)
df_full["voteStd"]      = df_full["voteStd"].fillna(0)

# ---------------------------------------
# 4. ALLENA MODELLO AI (Random Forest)
# ---------------------------------------
features = [
    "avgVote", "bestVote", "worstVote", "gamesPlayed", "votesReceived",
    "avgVoteVC", "sentimentAvg", "commentCount", "voteStd",
    "wins", "draws", "losses", "mvpCount", "hustleCount", "bestGoalCount", "winRate"
]

@st.cache_resource
def train_model(df):
    df_ml = df[df["gamesPlayed"] > 0].dropna(subset=features)
    X = df_ml[features]
    y = df_ml["avgVoteVC"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df_full)

# ---------------------------------------
# 5. PREDIZIONE PUNTEGGIO GIOCATORI
# ---------------------------------------
df_full = df_full.copy()
df_full["aiScore"] = model.predict(df_full[features].fillna(0))
df_full.loc[df_full["gamesPlayed"] == 0, "aiScore"] = df_full["aiScore"].min()

# ---------------------------------------
# 6. SELEZIONE GIOCATORI
# ---------------------------------------
all_names = df_full["displayName"].dropna().sort_values().tolist()
available = st.multiselect("Giocatori disponibili", all_names)

if len(available) < 4:
    st.warning("Seleziona almeno 4 giocatori.")
    st.stop()

df_selected  = df_full[df_full["displayName"].isin(available)].copy()
players_list = df_selected.to_dict("records")

# Ultimi 3 commenti per giocatore
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
# 7. ALGORITMO SQUADRE + RUOLI
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
# 8. GENERA SQUADRE AL CLICK
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

    def render_team(team, score):
        for p in team:
            sentiment_label = ""
            if p.get("sentimentAvg", 0) > 0.3:   sentiment_label = " 🔥"
            elif p.get("sentimentAvg", 0) < -0.1: sentiment_label = " 📉"
            mvp_tag      = " 🏆" * int(p.get("mvpCount", 0))
            hustle_tag   = " 💪" * int(p.get("hustleCount", 0))
            bestgoal_tag = " ⚽" * int(p.get("bestGoalCount", 0))
            st.write(
                f"• **{p['displayName']}** ({p.get('role','?')}) — "
                f"media {p.get('avgVoteVC', p.get('avgVote',0)):.2f} · "
                f"{int(p.get('wins',0))}V {int(p.get('draws',0))}P {int(p.get('losses',0))}S"
                f"{sentiment_label}{mvp_tag}{hustle_tag}{bestgoal_tag}"
            )
        st.metric("Forza totale", round(score, 2))

    col1, col2 = st.columns(2)
    with col1:
        st.header("⚪ Squadra Bianca")
        render_team(best["team1"], best["score1"])
    with col2:
        st.header("🔵 Squadra Colorata")
        render_team(best["team2"], best["score2"])

    diff_display = round(best["diff"], 2)
    if diff_display < 1:
        st.success(f"✅ Squadre molto equilibrate! Differenza: {diff_display}")
    else:
        st.info(f"ℹ️ Differenza di forza: {diff_display}")

    # ---------------------------------------
    # 9. DESCRIZIONE VERBALE VIA OLLAMA
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
                f"{int(p.get('wins',0))}V/{int(p.get('draws',0))}P/{int(p.get('losses',0))}S, "
                f"MVP {int(p.get('mvpCount',0))}x, hustle award {int(p.get('hustleCount',0))}x, "
                f"miglior gol {int(p.get('bestGoalCount',0))}x, "
                f"sentiment {'positivo' if p.get('sentimentAvg',0) > 0 else 'negativo' if p.get('sentimentAvg',0) < 0 else 'neutro'}"
                f"{comments_text}"
            )
        lines.append(f"Forza totale stimata: {score:.2f}")
        return "\n".join(lines)

    prompt = f"""Sei un commentatore sportivo di calcetto amatoriale.
Ti vengono fornite due squadre generate automaticamente da un algoritmo di bilanciamento.
Per ogni giocatore ricevi:
nome
ruolo(A=attaccante, C=centrocampista, D=difensore e P=portiere)
media voti
record vittorie/pareggi/sconfitte
numero di premi MVP e Hustle
sentiment dei commenti storici
ultimi commenti reali ricevuti
Scrivi una descrizione in italiano, tono informale e divertente, composta da 4-6 frasi, che rispetti queste regole:
Spiega chiaramente perché le squadre sono bilanciate, facendo riferimento a dati reali presenti nei profili (es. medie simili, distribuzione MVP, equilibrio nei record).
Cita almeno un dettaglio concreto proveniente dai commenti o dal record storico di un giocatore (es. un commento positivo/negativo, una serie di vittorie, un soprannome ricorrente).
Metti in evidenza i giocatori di punta di ciascuna squadra, scegliendoli in base ai dati forniti (miglior media, più MVP, record più solido).
Non inventare statistiche o informazioni non presenti nei dati. Usa solo ciò che è realmente fornito.
Mantieni uno stile leggero, ironico e tipico della telecronaca amatoriale.

SQUADRA BIANCA (⚪):
{format_team_for_prompt(best['team1'], best['score1'])}

SQUADRA COLORATA (🔵):
{format_team_for_prompt(best['team2'], best['score2'])}

Differenza di forza tra le squadre: {diff_display}
"""
    print(prompt)
    st.divider()
    st.subheader("🎙️ Commento AI")

    st.write("_Il modello sta generando il commento..._")
    output_box = st.empty()
    try:
        import requests, json as _json
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral-nemo", "prompt": prompt, "stream": True},
            stream=True,
            timeout=300
        )
        response.raise_for_status()
        narrative = ""
        for line in response.iter_lines():
            if line:
                chunk = _json.loads(line)
                narrative += chunk.get("response", "")
                output_box.markdown(narrative + "▌")
                if chunk.get("done", False):
                    break
        output_box.markdown(narrative)
    except requests.exceptions.ConnectionError:
        st.error("❌ Ollama non raggiungibile. Assicurati che sia avviato con `ollama serve`.")
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout superato (5 min). Il modello è troppo lento per questo hardware.")
    except Exception as e:
        st.error(f"❌ Errore durante la chiamata a Ollama: {e}")
