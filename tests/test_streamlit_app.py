from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from streamlit.testing.v1 import AppTest


PAGE_NAMES = [
    "overview",
    "data_explorer",
    "experiments",
    "predictions",
    "model_registry",
    "monitoring",
]


def _unavailable_data_sources(monkeypatch):
    from app import data_access

    monkeypatch.setattr(data_access, "load_churn_dataset", lambda: (None, "Dataset unavailable in test"))
    monkeypatch.setattr(data_access, "load_metrics_report", lambda: (None, "Report unavailable in test"))
    monkeypatch.setattr(data_access, "load_mlflow_snapshot", lambda: (None, "MLflow unavailable in test"))
    monkeypatch.setattr(data_access, "load_api_health", lambda: (None, "API unavailable in test"))
    monkeypatch.setattr(data_access, "check_prometheus", lambda: (False, "Prometheus unavailable in test"))
    monkeypatch.setattr(data_access, "query_prometheus", lambda _query: (None, "Prometheus unavailable in test"))
    monkeypatch.setattr(
        data_access,
        "query_prometheus_range",
        lambda _query, _start, _end, _step=60: (None, "Prometheus unavailable in test"),
    )


def test_navigation_entrypoint_renders_overview(monkeypatch):
    _unavailable_data_sources(monkeypatch)
    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"

    app = AppTest.from_file(app_path).run()

    assert not app.exception
    assert any("AutoML Studio" in title.value for title in app.title)


@pytest.mark.parametrize("page_name", PAGE_NAMES)
def test_each_workspace_page_renders_without_external_services(monkeypatch, page_name):
    _unavailable_data_sources(monkeypatch)
    page_path = Path(__file__).resolve().parents[1] / "app" / "app_pages" / f"{page_name}.py"

    app = AppTest.from_file(page_path).run()

    assert not app.exception


def test_dashboard_sends_prediction_to_api(monkeypatch):
    from app import data_access

    monkeypatch.setattr(data_access, "API_URL", "http://api-test")
    response = Mock()
    response.json.return_value = {
        "model": "test-model",
        "churn_probability": 0.6,
        "threshold": 0.35,
        "churn_prediction": 1,
        "risk_label": "High Risk (Churn)",
    }

    with patch("httpx.post", return_value=response) as post:
        app_path = Path(__file__).resolve().parents[1] / "app" / "app_pages" / "predictions.py"
        app = AppTest.from_file(app_path).run()
        app.button[0].click().run()

    assert not app.exception
    post.assert_called_once()
    assert post.call_args.args[0] == "http://api-test/predict"
    assert len(post.call_args.kwargs["json"]) == 19
    assert app.selectbox[0].label == "Genre"
    assert post.call_args.kwargs["json"]["gender"] == "Female"
    assert len(app.metric) == 1
    assert "60,0 %" in app.metric[0].value


@pytest.mark.parametrize("page_name", ["overview", "experiments"])
def test_model_comparison_charts_do_not_stack(monkeypatch, page_name):
    import streamlit as st
    from app import data_access

    _unavailable_data_sources(monkeypatch)
    sample_run = {
        "model": "GradientBoosting",
        "status": "FINISHED",
        "start_time": 1795269009848,
        "accuracy": 0.78,
        "precision": 0.56,
        "recall": 0.71,
        "f1_score": 0.63,
        "roc_auc": 0.84,
    }
    snapshot = {
        "candidate_runs": [sample_run],
        "run_history": [],
        "registry_versions": [],
        "total_runs": 1,
        "latest_run": None,
    }
    report = {
        "model": "GradientBoosting",
        "threshold": 0.35,
        "accuracy": 0.78,
        "precision": 0.56,
        "recall": 0.71,
        "f1_score": 0.63,
        "roc_auc": 0.84,
    }
    monkeypatch.setattr(data_access, "load_mlflow_snapshot", lambda: (snapshot, None))
    monkeypatch.setattr(data_access, "load_metrics_report", lambda: (report, None))
    chart = Mock()
    monkeypatch.setattr(st, "bar_chart", chart)
    page_path = Path(__file__).resolve().parents[1] / "app" / "app_pages" / f"{page_name}.py"

    app = AppTest.from_file(page_path).run()

    assert not app.exception
    chart.assert_called_once()
    assert chart.call_args.kwargs["stack"] is False


def test_monitoring_skips_non_finite_prometheus_samples(monkeypatch):
    import streamlit as st
    from app import data_access

    _unavailable_data_sources(monkeypatch)
    monkeypatch.setattr(
        data_access,
        "query_prometheus_range",
        lambda *_args, **_kwargs: (
            [{"values": [[1795269000, "NaN"], [1795269060, "Inf"]]}],
            None,
        ),
    )
    chart = Mock()
    monkeypatch.setattr(st, "line_chart", chart)
    page_path = Path(__file__).resolve().parents[1] / "app" / "app_pages" / "monitoring.py"

    app = AppTest.from_file(page_path).run()

    assert not app.exception
    chart.assert_not_called()
