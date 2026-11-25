from app.data_access import _normalize_metrics


def test_normalize_mlflow_validation_metrics_for_comparison():
    metrics = {
        "validation_f1_score": 0.6537,
        "validation_roc_auc": 0.8454,
        "f1_score": 0.5,
    }

    normalized = _normalize_metrics(metrics, prefer_validation=True)

    assert normalized["f1_score"] == 0.6537
    assert normalized["roc_auc"] == 0.8454
    assert normalized["validation_f1_score"] == 0.6537


def test_preserve_test_metrics_for_registered_model_history():
    metrics = {"f1_score": 0.6377, "validation_f1_score": 0.6537}

    normalized = _normalize_metrics(metrics, prefer_validation=False)

    assert normalized["f1_score"] == 0.6377
