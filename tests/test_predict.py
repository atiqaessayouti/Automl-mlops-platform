import numpy as np
import pandas as pd

from src import predict


def test_prediction_keeps_transformed_feature_names(monkeypatch):
    class Preprocessor:
        def transform(self, features):
            assert list(features.columns) == ["tenure"]
            return np.array([[12.0]])

        def get_feature_names_out(self):
            return np.array(["num__tenure"])

    class Model:
        def predict_proba(self, features):
            assert isinstance(features, pd.DataFrame)
            assert list(features.columns) == ["num__tenure"]
            return np.array([[0.7, 0.3]])

    monkeypatch.setattr(
        predict,
        "load_pipeline",
        lambda: {
            "model_name": "test-model",
            "preprocessor": Preprocessor(),
            "model": Model(),
            "threshold": 0.35,
        },
    )

    result = predict.predict_churn({"tenure": 12})

    assert result["churn_probability"] == 0.3
    assert result["risk_label"] == "Low Risk (Stay)"
