import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.train import tune_candidate


def test_optuna_tunes_candidate_using_stratified_folds():
    values, labels = make_classification(
        n_samples=120,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    features = pd.DataFrame(values, columns=[f"feature_{index}" for index in range(5)])
    target = pd.Series(labels)

    model, study = tune_candidate(
        "LogisticRegression",
        features,
        target,
        n_trials=2,
    )

    assert len(study.trials) == 2
    assert np.isfinite(study.best_value)
    assert model.get_params()["C"] == study.best_params["C"]
    assert model.get_params()["class_weight"] == (
        None if study.best_params["class_weight"] == "none" else "balanced"
    )
