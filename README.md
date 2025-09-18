# MLOps Assignment 1 — Iris Classification with MLflow

A clean, end‑to‑end example of experiment tracking and lightweight model management using **MLflow** on the classic **Iris** dataset.  
This repository includes reproducible training code, artifact logging (plots + serialized models), and a local Model Registry workflow.

---

## 📦 Project Structure

```
MLops-Assignment1/
├── data/                # (optional) raw/processed data
├── notebooks/           # exploratory notebooks (optional)
├── src/
│   └── train.py         # trains 3 models, logs to MLflow, saves .pkl, registers best
├── models/              # saved serialized models (.pkl)
├── results/             # plots and other artifacts (e.g., confusion matrices)
├── mlruns/              # MLflow tracking & artifacts (created at runtime)
├── mlflow.db            # SQLite backend for MLflow (created at runtime)
├── requirements.txt
└── README.md
```

---

## 🎯 Objectives

- Train and evaluate **three classifiers**: Logistic Regression, SVM (RBF), and Random Forest.  
- Track **parameters, metrics, and artifacts** with MLflow.  
- Save serialized models (`.pkl`) under `models/`.  
- Compare runs via **MLflow UI** and **register the best model** in the Model Registry.

---

## 🛠️ Getting Started (Windows + PowerShell)

> **Prerequisites**
> - Python 3.10+ installed and available as `python`
> - Git installed
> - VS Code (recommended)

### 1) Create & activate a virtual environment
From the project root:
```powershell
python -m venv .venv

# If activation is blocked by policy
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate
. .\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> If you don’t have `requirements.txt`, install directly:
> ```powershell
> pip install mlflow scikit-learn matplotlib pandas joblib
> ```

---

## 🚀 Run MLflow Tracking Server (local)

Keep this terminal open while training:
```powershell
python -m mlflow server `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root ./mlruns `
  --host 127.0.0.1 `
  --port 5000
```
Open the UI in your browser: **http://127.0.0.1:5000**

---

## 🧪 Train Models & Log Runs

Open a **new** VS Code terminal (the venv remains active) and set the tracking URI:
```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python .\src\train.py
```

What happens:
- Trains **LogReg**, **SVM (RBF)**, **RandomForest**
- Logs params/metrics/artifacts to MLflow
- Saves each model to `models/<name>.pkl`
- Saves confusion matrices to `results/`
- **Auto‑registers** the best model (by macro‑F1) as `IrisClassifier`

You can verify in MLflow UI:
- **Experiments** → `mlops-assignment-1` → compare runs
- **Models** → `IrisClassifier` → view versions & source runs

---

## 📈 Metrics & Artifacts

For each run, the following are logged:
- **Metrics:** `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`
- **Artifacts:** confusion matrix image, serialized `.pkl` model
- **Model artifact:** scikit‑learn pipeline logged via `mlflow.sklearn.log_model`

Artifacts are accessible from each run page in the UI, and locally under `results/` and `models/`.

---

## 🔁 Reproducibility

- A fixed `random_state` is used where relevant.
- To adjust hyperparameters, edit `src/train.py` (the three candidate estimators section) and rerun training.
- Use MLflow UI to compare before/after.

---

## 📤 GitHub Workflow

Make small, frequent commits with clear messages:
```powershell
git add .
git commit -m "feat: add MLflow training & logging"
git push origin main
```
If you created empty folders, ensure they contain a placeholder (e.g., `.gitkeep`) so Git tracks them.

---
# Screen Shots
<img width="600" height="600" alt="confusion_svm_rbf" src="https://github.com/user-attachments/assets/a4d74d98-0581-4730-84ad-477b4c043553" />
<img width="600" height="600" alt="confusion_rf" src="https://github.com/user-attachments/assets/abe601f8-4dbb-4493-92eb-f9dee037dde8" />
<img width="600" height="600" alt="confusion_logreg" src="https://github.com/user-attachments/assets/f53b3234-f7bf-4b20-bf72-5ca5246cedf3" />
<img width="1919" height="1003" alt="Experiment Run" src="https://github.com/user-attachments/assets/08ab5538-ee7b-47e5-9d3f-e8ad5be6590c" />
<img width="1919" height="1008" alt="Model Registery" src="https://github.com/user-attachments/assets/5621fb3c-7b66-453e-9efa-62b2b5e8dc80" />





## 🧰 Troubleshooting

**Execution policy prevents venv activation**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
```

**'mlflow' not recognized**
- Ensure the venv is active (prompt shows `(.venv)`).
- Use the module form: `python -m mlflow server ...`

**403 when pushing to GitHub**
- Confirm your remote points to the correct account/repo:
  ```powershell
  git remote -v
  git remote set-url origin https://github.com/<your-username>/MLops-Assignment1.git
  ```
- Clear cached credentials in Windows Credential Manager and push again (use a Personal Access Token).

---

## 📚 Notes

- Dataset: **Iris** (three classes: setosa, versicolor, virginica).  
- Purpose: demonstrate experiment tracking, artifact logging, and a simple Model Registry flow.  
- This project uses a **local** MLflow server with SQLite; switch to a hosted tracking server or cloud artifact store for team scenarios.

---

## 🙌 Acknowledgements

- MLflow (https://mlflow.org/)
- scikit‑learn (https://scikit-learn.org/)
- UCI Iris Dataset / Fisher’s Iris
