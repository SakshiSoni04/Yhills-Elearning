# src/utils.py
import pandas as pd

def load_courses(path: str) -> pd.DataFrame:
    """
    Load courses dataset from CSV file.
    Ensures correct dtypes and missing values handled.
    """
    df = pd.read_csv(path)

    # Ensure correct column names exist
    expected_cols = [
        "Subject", "Title", "Institution", "Learning Product",
        "Level", "Duration", "Gained Skills", "Rate", "Reviews"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Clean up data
    df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce").fillna(0)
    df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce").fillna(0).astype(int)

    df["Gained Skills"] = df["Gained Skills"].fillna("")

    return df