# ⚽ Generatore di Squadre con AI

Applicazione web per generare squadre di calcetto bilanciate usando un modello di Machine Learning (Random Forest). L'AI analizza le statistiche dei giocatori e divide automaticamente il gruppo in due squadre equilibrate per ruolo e punteggio.

---

## Requisiti

- Python 3.9+
- pip

---

## Installazione

### 1. Clona il repository

```bash
git clone https://github.com/tuo-utente/generatore-squadre.git
cd generatore-squadre
```

### 2. (Opzionale) Crea un ambiente virtuale

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

Se non hai un `requirements.txt`, installa manualmente:

```bash
pip install streamlit pandas numpy scikit-learn
```

---

## Struttura del progetto

```
generatore-squadre/
├── app.py              # Applicazione principale Streamlit
├── players.csv         # Anagrafica giocatori (id, name, role)
├── stats.csv           # Statistiche giocatori (playerId, avgVote, goals, ...)
├── requirements.txt    # Dipendenze Python
└── README.md
```

---

## Formato dei CSV

### `players.csv`

| Campo | Tipo   | Descrizione              |
|-------|--------|--------------------------|
| id    | int    | Identificatore univoco   |
| name  | string | Nome del giocatore       |
| role  | string | Ruolo: P, D, C, A        |

```csv
id,name,role
1,Marco Rossi,P
2,Luca Bianchi,D
```

### `stats.csv`

| Campo     | Tipo  | Descrizione                     |
|-----------|-------|---------------------------------|
| playerId  | int   | Riferimento a `players.csv`     |
| avgVote   | float | Voto medio (es. 6.8)            |
| goals     | int   | Gol segnati                     |
| assists   | int   | Assist effettuati               |
| bestVote  | float | Voto migliore                   |
| worstVote | float | Voto peggiore                   |
| games     | int   | Partite giocate                 |

```csv
playerId,avgVote,goals,assists,bestVote,worstVote,games
1,6.5,0,1,7.5,5.5,10
2,7.1,2,3,8.0,6.0,12
```

---

## Avvio

```bash
streamlit run app.py
```

L'app sarà disponibile nel browser all'indirizzo:

```
http://localhost:8501
```

---

## Come funziona

1. **Carica i dati** — legge `players.csv` e `stats.csv` e li unisce tramite `playerId`.
2. **Allena il modello** — addestra un Random Forest (200 alberi) sulle statistiche per calcolare un `aiScore` per ogni giocatore.
3. **Selezione giocatori** — scegli i giocatori disponibili per la partita tramite il multiselect.
4. **Bilanciamento squadre** — l'algoritmo testa tutte le combinazioni possibili e minimizza la differenza di forza tra le due squadre, applicando una penalità se mancano ruoli fondamentali (D, C, A).
5. **Risultati** — vengono mostrate le due squadre con il punteggio AI di ogni giocatore e il totale.

---

## Generare `requirements.txt`

Se vuoi creare il file delle dipendenze dal tuo ambiente attuale:

```bash
pip freeze > requirements.txt
```

Oppure crea un file minimale:

```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

---

## Licenza

MIT
