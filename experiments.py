import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from ensemble_models import SELECTED_DEPTH, metrics, model, run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput


def main():
    root = Path(__file__).resolve().parent
    frame = pd.read_csv(root / "german.csv", sep=";")
    target = frame.pop("Creditability")
    x, xt, y, yt = train_test_split(frame, target, test_size=0.2, stratify=target, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = {}
    for depth in (1, 2, 3):
        p = cross_val_predict(model(depth, 42), x, y, cv=cv, method="predict_proba")[:, 1]
        scores[depth] = metrics(y, p, (p > 0.5).astype(int))
    best = max(scores, key=lambda d: scores[d]["roc_auc"])
    print(scores, "best", best)
    if best != SELECTED_DEPTH:
        raise ValueError("Selected depth differs from the training CV winner")
    result = run_ensemble_experiment(EnsembleInput(x, y, xt, yt))
    report = dict(seed=42, train_rows=len(y), test_rows=len(yt), selected_depth=best,
                  training_cv=scores, test={k: getattr(result, k) for k in scores[best]})
    (root / "experiment_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
