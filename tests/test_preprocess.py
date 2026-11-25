import joblib
import pandas as pd

from src.preprocess import preprocess_dataset


def test_preprocessing_splits_encodes_and_persists(tmp_path):
    rows = 60
    raw_data = pd.DataFrame(
        {
            "customerID": [f"customer-{index}" for index in range(rows)],
            "gender": ["Female", "Male"] * (rows // 2),
            "tenure": list(range(rows)),
            "MonthlyCharges": [20.0 + index for index in range(rows)],
            "TotalCharges": [
                "" if index % 7 == 0 else str(20.0 * index)
                for index in range(rows)
            ],
            "Contract": ["Month-to-month", "One year", "Two year"] * (rows // 3),
            "Churn": ["Yes" if index % 4 == 0 else "No" for index in range(rows)],
        }
    )
    raw_path = tmp_path / "input.csv"
    processed_dir = tmp_path / "processed"
    preprocessor_path = tmp_path / "models" / "preprocessor.joblib"
    raw_data.to_csv(raw_path, index=False)

    result = preprocess_dataset(raw_path, processed_dir, preprocessor_path)

    X_train = pd.read_csv(processed_dir / "X_train_processed.csv")
    X_test = pd.read_csv(processed_dir / "X_test_processed.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv")["Churn"]
    y_test = pd.read_csv(processed_dir / "y_test.csv")["Churn"]
    cleaned_data = pd.read_csv(processed_dir / "telco_churn_clean.csv")
    preprocessor = joblib.load(preprocessor_path)

    assert result == {
        "train_rows": 48,
        "test_rows": 12,
        "feature_count": X_train.shape[1],
        "target_values": [0, 1],
    }
    assert not X_train.empty and not X_test.empty
    assert X_train.columns.equals(X_test.columns)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    assert preprocessor.transform(
        cleaned_data.drop(columns=["Churn", "customerID"]).iloc[:1]
    ).shape[1] == X_train.shape[1]
    assert (processed_dir / "telco_churn_clean.csv").is_file()
