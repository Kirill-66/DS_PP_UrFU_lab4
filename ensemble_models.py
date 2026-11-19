from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def model(depth, seed):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingClassifier(n_estimators=180, learning_rate=0.04,
                                   max_depth=depth, random_state=seed),
    )


def metrics(y, probabilities, predictions):
    return dict(roc_auc=roc_auc_score(y, probabilities), accuracy=accuracy_score(y, predictions),
                precision=precision_score(y, predictions, zero_division=0),
                recall=recall_score(y, predictions, zero_division=0),
                f1=f1_score(y, predictions, zero_division=0))


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    fitted = model(SELECTED_DEPTH, data.random_seed).fit(data.train_features, data.train_target)
    probabilities = fitted.predict_proba(data.test_features)[:, 1]
    predictions = fitted.predict(data.test_features)
    return EnsembleResult(f"Gradient Boosting (depth={SELECTED_DEPTH})", probabilities, predictions,
                          **metrics(data.test_target, probabilities, predictions))


SELECTED_DEPTH = 3
