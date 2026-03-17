# ⚽ AI Squad Maker

> Divide un gruppo di giocatori in due squadre equilibrate usando Machine Learning locale.

---

## 📋 Descrizione

AI Squad Maker analizza i dati storici delle partite (voti, presenze, performance) e allena un modello ML per calcolare la forza di ogni giocatore. Selezionati i presenti della serata, l'algoritmo trova la divisione ottimale bilanciando livello individuale e ruoli in campo. Tutto gira in locale — nessun dato inviato online.

---

## 🗂 Struttura dei file

```
AIsquadreMaker/
├── app.py
├── players.csv
├── stats.csv
└── matches.csv
```

### players.csv
| Campo | Tipo | Descrizione |
|---|---|---|
| `id` | string | ID univoco del giocatore (es. `p01`) |
| `name` | string | Nome del giocatore |
| `role` | string | Ruolo: `P`, `D`, `C`, `A` |
| `icon` | string | Nome icona |
| `imagePath` | string | Percorso immagine profilo |

### stats.csv
| Campo | Tipo | Descrizione |
|---|---|---|
| `playerId` | string | Riferimento a `players.id` |
| `playerName` | string | Nome del giocatore |
| `role` | string | Ruolo |
| `gamesPlayed` | int | Partite giocate |
| `votesReceived` | int | Voti ricevuti |
| `avgVote` | float | Media voti |
| `bestVote` | float | Voto migliore |
| `worstVote` | float | Voto peggiore |

### matches.csv
| Campo | Tipo | Descrizione |
|---|---|---|
| `id` | string | ID partita |
| `date` | string | Data e ora |
| `fieldLocation` | string | Campo da gioco |
| `scoreA` / `scoreB` | int | Punteggio squadre |
| `teamA` / `teamB` | string | Giocatori delle squadre |
| `mvp` | string | MVP della partita |
| `hustlePlayer` | string | Giocatore più combattivo |
| `bestGoalPlayer` | string | Autore del gol più bello |

---

## 🚀 Installazione

### Requisiti

- Python 3.9+
- pip

### Dipendenze

```bash
pip install streamlit pandas numpy scikit-learn
```

### Avvio

```bash
streamlit run app.py
```

---

## 🧠 Come funziona

1. **Caricamento dati** — vengono letti `players.csv` e `stats.csv` e uniti tramite `playerId`
2. **Training del modello** — un `RandomForestRegressor` viene addestrato sui giocatori con almeno una partita giocata, usando come feature `avgVote`, `bestVote`, `worstVote`, `gamesPlayed`, `votesReceived`
3. **Calcolo aiScore** — ogni giocatore riceve un punteggio di forza predetto dal modello; chi non ha mai giocato riceve il punteggio minimo
4. **Selezione** — si scelgono i giocatori presenti tramite il multiselect
5. **Ottimizzazione** — l'algoritmo esplora tutte le combinazioni possibili di divisione e trova quella con la minima differenza di forza tra le due squadre, con una penalità aggiuntiva se una squadra manca di ruoli fondamentali (D, C, A)

---

## 🎮 Utilizzo

1. Avvia l'app con `streamlit run app.py`
2. Seleziona i giocatori presenti dal menu a tendina
3. Clicca **⚽ Genera Squadre**
4. Visualizza le due squadre con forza totale e differenza di equilibrio

---

## ⚙️ Ruoli supportati

| Codice | Ruolo |
|---|---|
| `P` | Portiere |
| `D` | Difensore |
| `C` | Centrocampista |
| `A` | Attaccante |

---

## 📌 Note

- Il modello viene ricalcolato solo al primo avvio grazie a `@st.cache_resource`
- Con molti giocatori (es. 20+) il calcolo delle combinazioni può richiedere qualche secondo
- I giocatori con `gamesPlayed = 0` sono selezionabili ma ricevono il punteggio minimo
