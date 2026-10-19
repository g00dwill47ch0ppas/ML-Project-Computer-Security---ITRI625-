# ITRI625 – Phishing E-mail Detection (BiLSTM)

Binary classifier (phishing vs legitimate e-mail) built in PyTorch, served through FastAPI and used from a Tkinter desktop app. Includes LIME explanations (XAI).

## Files
| File | Purpose |
|---|---|
| `ITRI625_Phishing_Detection.ipynb` | Full pipeline: cleaning, split, baseline, BiLSTM, early stopping, metrics, graphs, XAI |
| `phishing_core.py` | Tokeniser, model class, inference wrapper (written by the notebook, shared with the API) |
| `api.py` | FastAPI service: `/health`, `/predict`, `/explain` |
| `gui.py` | Tkinter desktop client |
| `artifacts/` | Created by the notebook: `model.pt`, `vocab.json`, `config.json`, `training_history.csv` |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Download **Phishing_Email.csv** from Kaggle ("Phishing Email Dataset") into this folder.

## Run
1. Open the notebook → *Restart & Run All* (creates `artifacts/`).
2. `uvicorn api:app` (docs at http://127.0.0.1:8000/docs)
3. `python gui.py`

## Results

