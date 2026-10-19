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

## Results (held-out test set, 2,629 e-mails)
 
| Metric | TF-IDF + Logistic Regression (baseline) | **BiLSTM** |
|---|---|---|
| Accuracy | 0.9840 | 0.9700 |
| Precision | 0.9796 | 0.9475 |
| Recall | 0.9776 | 0.9735 |
| F1-score | 0.9786 | 0.9603 |
| AUC-ROC | 0.9982 | 0.9938 |
| Loss | 0.0836 | 0.0963 |
 
BiLSTM confusion matrix: **956** TP · **1,594** TN · **53** FP · **26** FN (false-positive rate 3.22%, false-negative rate 2.65%).
 
**Honest finding:** the simple TF-IDF baseline beat the BiLSTM on every test metric. On this dataset many cues are word-level (product and offer vocabulary), which a linear model on word and bigram counts already separates very well. The BiLSTM also took ~158 minutes to train on a CPU. LIME further suggests the legitimate class is dominated by one corporate mail style, so scores probably overestimate real-world performance (see *Limitations*).
 
| Training history | ROC / PR curves |
|---|---|
| ![Training history](docs/figures/figure1_training_curves.png) | ![ROC and PR](docs/figures/figure2_roc_pr.png) |
| **Confusion matrix and score distribution** | **LIME explanations** |
| ![Confusion matrix](docs/figures/figure3_confusion.png) | ![LIME](docs/figures/figure4_lime.png) |
 
---
 
## How it works
 
```
Phishing_Email.csv
   │  clean: drop missing/empty, conflicting labels, duplicates  (18,650 → 17,521)
   ▼
stratified split 70 / 15 / 15  (12,264 train · 2,628 val · 2,629 test)
   │
   ├─► TF-IDF + Logistic Regression ........ baseline
   │
   └─► tokenise (URLs → <url>, addresses → <email>) → vocab from TRAIN only (30,000)
        → pad/truncate to 256 tokens
        → Embedding(128) → Dropout → BiLSTM(128×2) → [mean-pool ‖ max-pool]
        → Dropout → Dense(64, ReLU) → Dropout → Dense(1) → sigmoid = P(phishing)
        │
        ├─ Adam (lr 1e-3), BCE loss, gradient clipping, ReduceLROnPlateau
        ├─ per-epoch metrics logged (loss, accuracy, precision, recall, F1, AUC-ROC)
        └─ early stopping: patience 4 on validation loss, best weights restored (epoch 4 of 8)
              │
              ▼
        artifacts/ (model.pt, vocab.json, config.json)  →  FastAPI  →  Tkinter app (+ LIME)
```
 
**Model size:** 4,137,089 trainable parameters.
 
---
 
## Repository structure
 
| Path | Purpose |
|---|---|
| `ITRI625_Phishing_Detection.ipynb` | Full pipeline: cleaning, EDA, split, baseline, BiLSTM, early stopping, metrics, graphs, XAI |
| `phishing_core.py` | Tokeniser, vocabulary, `PhishingBiLSTM`, `PhishingClassifier` (shared by notebook and API) |
| `api.py` | FastAPI service: `GET /health`, `POST /predict`, `POST /explain` |
| `gui.py` | Tkinter desktop client |
| `artifacts/` | `model.pt`, `vocab.json`, `config.json`, `training_history.csv`, `test_metrics.json` (created by the notebook) |
| `demo_emails/` | Sample e-mails for trying the app |
| `docs/figures/` | Plots used in this README |
| `requirements.txt`, `.gitignore` | Dependencies; the dataset is not committed |
 
---
 
## Setup and run
 
```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
 
1. Download `Phishing_Email.csv` from Kaggle ([Phishing Email Detection](https://www.kaggle.com/datasets/subhajournal/phishingemails)) into the project folder.
2. Open the notebook, select the `.venv` kernel and **Run All**. This creates `artifacts/`.
3. Start the API (terminal 1): `uvicorn api:app`
4. Start the desktop app (terminal 2): `python gui.py`
### Using the desktop app
Paste an e-mail (or **Load file…**) and click **Analyse** for the verdict and phishing probability. Click **Explain** to highlight the words behind the decision: **red** pushes towards phishing, **green** towards legitimate.
 
### API
Interactive docs at `http://127.0.0.1:8000/docs`.
 
```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text": "Your e-mail text here"}'
```
 
Response shape: `{"label": "phishing", "is_phishing": true, "probability": 0.97, "threshold": 0.5}`
 
| Endpoint | Purpose |
|---|---|
| `GET /health` | Confirms the model is loaded |
| `POST /predict` | Verdict and probability |
| `POST /explain` | LIME word weights (positive = towards phishing) |
 
---
 
## Design decisions
 
- **Duplicates and conflicting labels removed before splitting**, so the same e-mail cannot appear in train and test.
- **Vocabulary built from the training set only**, to avoid leakage.
- **Test set used once**, after training and early stopping were finished.
- **Baseline model** included to check whether the deep network is actually worth its cost.
- **Shared `phishing_core.py`** so training and serving use identical preprocessing.
- **Early stopping on validation loss** (smooth and threshold-independent) with the best weights restored.
## Limitations
 
- Training data come from fixed historic collections, so concept drift is likely with newer phishing campaigns.
- The data mix credential phishing with advert-style spam, and the legitimate class appears to be dominated by one corporate mail style, so the model may partly learn source or topic.
- Text only: no headers, sender reputation or attachments; e-mails are truncated at 256 tokens.
- Not tested against deliberately adversarial e-mails.
- Embeddings are trained from scratch (no pre-trained vectors).
## Reproducibility
 
Random seed 42 (Python, NumPy, PyTorch). Hyperparameters are set in the notebook's configuration cell. Exact results can vary slightly across hardware.

