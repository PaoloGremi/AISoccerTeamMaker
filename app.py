import streamlit as st
import pandas as pd
import numpy as np
import itertools
import re
import requests
import json as _json
import warnings
warnings.filterwarnings("ignore", category=Warning, module="urllib3")
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Generatore di Squadre", page_icon="⚽", layout="centered")

# ---------------------------------------
# NAVIGAZIONE
# ---------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page
    st.rerun()

# ---------------------------------------
# 1. CARICA I CSV
# ---------------------------------------
@st.cache_data
def load_data():
    players = pd.read_csv("players.csv")
    stats   = pd.read_csv("stats.csv")
    vc      = pd.read_csv("votes_comments.csv")
    matches = pd.read_csv("matches.csv")
    return players, stats, vc, matches

players, stats, vc, matches = load_data()

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
    if sA > sB:   resA, resB = "W", "L"
    elif sA < sB: resA, resB = "L", "W"
    else:          resA, resB = "D", "D"
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

# Ultimi 3 commenti per giocatore (calcolato una volta sola)
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
# 4. ALLENA MODELLO AI (Random Forest)
# ---------------------------------------
FEATURES = [
    "avgVote", "bestVote", "worstVote", "gamesPlayed", "votesReceived",
    "avgVoteVC", "sentimentAvg", "commentCount", "voteStd",
    "wins", "draws", "losses", "mvpCount", "hustleCount", "bestGoalCount", "winRate"
]

@st.cache_resource
def train_model(df):
    df_ml = df[df["gamesPlayed"] > 0].dropna(subset=FEATURES)
    X = df_ml[FEATURES]
    y = df_ml["avgVoteVC"]
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model

model = train_model(df_full)

df_full["aiScore"] = model.predict(df_full[FEATURES].fillna(0))
df_full.loc[df_full["gamesPlayed"] == 0, "aiScore"] = df_full["aiScore"].min()

# ---------------------------------------
# HELPERS CONDIVISI
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
    sum1   = sum(p["aiScore"] for p in team1)
    sum2   = sum(p["aiScore"] for p in team2)
    roles1 = [p.get("role", "") for p in team1]
    roles2 = [p.get("role", "") for p in team2]
    penalty = role_penalty(roles1) + role_penalty(roles2)
    return abs(sum1 - sum2) + penalty, team1, team2, sum1, sum2

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

def call_ollama(prompt):
    output_box = st.empty()
    try:
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

def build_prompt_genera(team1, score1, team2, score2, diff_display):
    return f"""Sei un commentatore sportivo di calcetto amatoriale.
Ti vengono fornite due squadre generate automaticamente da un algoritmo di bilanciamento.
Per ogni giocatore ricevi:
nome, ruolo (A=attaccante, C=centrocampista, D=difensore, P=portiere),
media voti, record vittorie/pareggi/sconfitte, premi MVP e Hustle,
sentiment dei commenti storici, ultimi commenti reali ricevuti.

Scrivi una descrizione in italiano, tono informale e divertente, composta da 4-6 frasi:
- Spiega perché le squadre sono bilanciate citando dati reali (medie, MVP, record).
- Cita almeno un dettaglio concreto dai commenti o dal record storico di un giocatore.
- Metti in evidenza i giocatori di punta di ciascuna squadra.
- Non inventare statistiche non presenti nei dati.
- Stile leggero, ironico, da telecronaca amatoriale.

SQUADRA BIANCA (⚪):
{format_team_for_prompt(team1, score1)}

SQUADRA COLORATA (🔵):
{format_team_for_prompt(team2, score2)}

Differenza di forza tra le squadre: {diff_display}
"""

def build_prompt_telecronaca(team1, team2):
    return f"""Sei un telecronista sportivo italiano appassionato ed esperto.
Stai per commentare una partita di calcetto tra amici.
Scrivi un commento di presentazione in stile telecronaca:
- Incipit entusiasmante che presenti la sfida.
- Presenta la Squadra 1 con una breve frase personale per ogni giocatore,
  basandoti sui dati reali (media voti, record, MVP, commenti storici).
- Presenta la Squadra 2 allo stesso modo.
- Chiudi con una previsione sul match e un incitamento ai tifosi.

Tono: appassionato, divertente, enfatico come i grandi telecronisti italiani.
Lunghezza: circa 250-300 parole. Non inventare statistiche non presenti nei dati.

SQUADRA 1:
{format_team_for_prompt(team1, sum(p["aiScore"] for p in team1))}

SQUADRA 2:
{format_team_for_prompt(team2, sum(p["aiScore"] for p in team2))}

Inizia subito con la telecronaca, senza premesse."""


# ======================================
# PAGINA: HOME
# ======================================
def page_home():
    st.title("⚽ Generatore di Squadre")
    st.markdown("### Come vuoi procedere?")
    st.markdown(" ")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="border:1px solid #ddd; border-radius:10px; padding:1.2rem; height:160px; display:flex; flex-direction:column; justify-content:center;">
                <h4 style="margin:0 0 0.5rem 0;">🎲 Non ho ancora le squadre</h4>
                <p style="font-size:14px; color:gray; margin:0;">
                    Lascia all'AI il compito di bilanciare i giocatori disponibili
                    in due squadre equilibrate per ruolo e punteggio.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div style="margin-top:0.6rem;"></div>', unsafe_allow_html=True)
        if st.button("Genera le squadre", use_container_width=True):
            go_to("genera")

    with col2:
        st.markdown(
            """
            <div style="border:1px solid #ddd; border-radius:10px; padding:1.2rem; height:160px; display:flex; flex-direction:column; justify-content:center;">
                <h4 style="margin:0 0 0.5rem 0;">🎙️ Ho già scelto le squadre</h4>
                <p style="font-size:14px; color:gray; margin:0;">
                    Inserisci i giocatori di ogni squadra e ottieni un commento
                    da telecronista per presentare la partita.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div style="margin-top:0.6rem;"></div>', unsafe_allow_html=True)
        if st.button("Genera commento Pre-Partita", use_container_width=True):
            go_to("telecronaca")


# ======================================
# PAGINA: GENERA SQUADRE
# ======================================
def page_genera():
    st.title("⚽ Generatore di Squadre con AI Locale")

    if st.button("← Torna alla home"):
        go_to("home")

    st.markdown("---")

    all_names = df_full["displayName"].dropna().sort_values().tolist()
    available = st.multiselect("Giocatori disponibili", all_names)

    if len(available) < 4:
        st.warning("Seleziona almeno 4 giocatori.")
        st.stop()

    df_selected  = df_full[df_full["displayName"].isin(available)].copy()
    players_list = df_selected.to_dict("records")

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

        st.divider()
        st.subheader("🎙️ Commento AI")
        st.write("_Il modello sta generando il commento..._")
        prompt = build_prompt_genera(best["team1"], best["score1"], best["team2"], best["score2"], diff_display)
        print(prompt)
        call_ollama(prompt)


# ======================================
# PAGINA: TELECRONACA
# ======================================
def page_telecronaca():
    st.title("🎙️ Commento da Telecronista")

    if st.button("← Torna alla home"):
        go_to("home")

    st.markdown("---")

    all_names = df_full["displayName"].dropna().sort_values().tolist()

    st.subheader("Squadra 1")
    team1_names = st.multiselect(
        "Seleziona i giocatori della Squadra 1",
        options=all_names,
        key="team1"
    )

    st.subheader("Squadra 2")
    # I giocatori già scelti per la Squadra 1 non compaiono nella Squadra 2
    remaining = [n for n in all_names if n not in team1_names]
    team2_names = st.multiselect(
        "Seleziona i giocatori della Squadra 2",
        options=remaining,
        key="team2"
    )

    if team1_names and len(team1_names) < 2:
        st.warning("Aggiungi almeno 2 giocatori alla Squadra 1.")
    if team2_names and len(team2_names) < 2:
        st.warning("Aggiungi almeno 2 giocatori alla Squadra 2.")

    ready = len(team1_names) >= 2 and len(team2_names) >= 2

    if ready:
        team1_players = df_full[df_full["displayName"].isin(team1_names)].to_dict("records")
        team2_players = df_full[df_full["displayName"].isin(team2_names)].to_dict("records")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Formazione Squadra 1**")
            render_team(team1_players, sum(p["aiScore"] for p in team1_players))
        with col2:
            st.markdown("**Formazione Squadra 2**")
            render_team(team2_players, sum(p["aiScore"] for p in team2_players))

        st.markdown("---")

        if st.button("🎙️ Genera commento telecronista (AI Locale)"):
            st.subheader("📻 In diretta dallo stadio...")
            prompt = build_prompt_telecronaca(team1_players, team2_players)
            call_ollama(prompt)


# ======================================
# ROUTER
# ======================================
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "genera":
    page_genera()
elif st.session_state.page == "telecronaca":
    page_telecronaca()
