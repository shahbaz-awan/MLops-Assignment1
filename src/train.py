import os, json
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# --------- CONFIG ----------
EXPERIMENT_NAME = "mlops-assignment-1"
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Use running server if provided; otherwise assume local MLflow server on 5000
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment(EXPERIMENT_NAME)

def plot_confusion(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def train_and_log(model_name, estimator, X_train, y_train, X_test, y_test):
    # Build a simple pipeline with scaling where it helps
    use_scaler = not isinstance(estimator, RandomForestClassifier)
    steps = [("scaler", StandardScaler())] if use_scaler else []
    steps.append(("clf", estimator))
    pipe = Pipeline(steps)

    with mlflow.start_run(run_name=model_name) as run:
        # Log key hyperparameters
        params = {k: v for k, v in estimator.get_params().items()
                  if isinstance(v, (int, float, str, bool, type(None)))}
        # Avoid logging huge objects
        for k,v in list(params.items()):
            if len(str(v)) > 200:
                params.pop(k)
        mlflow.log_params(params)

        # Train & predict
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("precision_macro", float(precision))
        mlflow.log_metric("recall_macro", float(recall))
        mlflow.log_metric("f1_macro", float(f1))

        # Confusion matrix as artifact
        cm = confusion_matrix(y_test, y_pred)
        cm_path = RESULTS_DIR / f"confusion_{model_name}.png"
        plot_confusion(cm, [str(i) for i in np.unique(y_test)],
                       f"Confusion Matrix - {model_name}", cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="artifacts")

        # Log model to MLflow and also save a .pkl locally
        mlflow.sklearn.log_model(sk_model=pipe, artifact_path="model")
        pkl_path = MODELS_DIR / f"{model_name}.pkl"
        joblib.dump(pipe, pkl_path)
        mlflow.log_artifact(str(pkl_path), artifact_path="artifacts")

        run_id = run.info.run_id
        return {
            "model_name": model_name,
            "run_id": run_id,
            "metrics": {"accuracy": acc, "precision_macro": precision,
                        "recall_macro": recall, "f1_macro": f1}
        }

def main():
    # Data
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    # Models
    candidates = [
        ("logreg", LogisticRegression(max_iter=1000, n_jobs=None)),
        ("svm_rbf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
    ]

    results = []
    for name, est in candidates:
        results.append(train_and_log(name, est, X_train, y_train, X_test, y_test))

    # Pick best by F1 (macro)
    best = max(results, key=lambda r: r["metrics"]["f1_macro"])
    print("RESULTS:", json.dumps(results, indent=2))
    print(f"BEST_MODEL={best['model_name']}  RUN_ID={best['run_id']}  F1={best['metrics']['f1_macro']:.4f}")

    # Register best model (requires MLflow server with SQL backend)
    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(model_uri=model_uri, name="IrisClassifier")
    print("REGISTERED_MODEL:", registered.name, "VERSION:", registered.version)

if __name__ == "__main__":
    main()
