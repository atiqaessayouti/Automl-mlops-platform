from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def preprocess_dataset(
    raw_data_path=RAW_DATA_PATH,
    processed_dir=PROCESSED_DIR,
    preprocessor_path=PREPROCESSOR_PATH,
):
    raw_data_path = Path(raw_data_path)
    processed_dir = Path(processed_dir)
    preprocessor_path = Path(preprocessor_path)

    if not raw_data_path.is_file():
        raise FileNotFoundError(f"Dataset brut introuvable : {raw_data_path}")

    data = pd.read_csv(raw_data_path)
    required_columns = {"customerID", "TotalCharges", "Churn"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Colonnes requises absentes du dataset : {sorted(missing_columns)}"
        )

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"].astype(str).str.strip(),
        errors="coerce",
    )
    data = data.drop_duplicates().reset_index(drop=True)

    target = data["Churn"].map({"No": 0, "Yes": 1})
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("La cible Churn doit contenir uniquement 'No' ou 'Yes'.")
    target = target.astype(int)
    features = data.drop(columns=["Churn", "customerID"])

    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = features.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()
    unhandled_columns = set(features.columns).difference(
        numeric_features + categorical_features
    )
    if unhandled_columns:
        raise ValueError(
            f"Types de variables non pris en charge : {sorted(unhandled_columns)}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    X_train_array = preprocessor.fit_transform(X_train)
    X_test_array = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    X_train_processed = pd.DataFrame(X_train_array, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_array, columns=feature_names)

    for name, matrix in (
        ("train", X_train_processed),
        ("test", X_test_processed),
    ):
        if matrix.empty or not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise ValueError(f"La matrice prétraitée {name} est vide ou non finie.")

    processed_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(processed_dir / "telco_churn_clean.csv", index=False)
    X_train_processed.to_csv(processed_dir / "X_train_processed.csv", index=False)
    X_test_processed.to_csv(processed_dir / "X_test_processed.csv", index=False)
    y_train.to_frame(name="Churn").to_csv(
        processed_dir / "y_train.csv",
        index=False,
    )
    y_test.to_frame(name="Churn").to_csv(
        processed_dir / "y_test.csv",
        index=False,
    )
    joblib.dump(preprocessor, preprocessor_path)

    return {
        "train_rows": len(X_train_processed),
        "test_rows": len(X_test_processed),
        "feature_count": X_train_processed.shape[1],
        "target_values": sorted(target.unique().tolist()),
    }


if __name__ == "__main__":
    result = preprocess_dataset()
    print(
        "Prétraitement terminé : "
        f"{result['train_rows']} lignes train, "
        f"{result['test_rows']} lignes test, "
        f"{result['feature_count']} variables."
    )
