from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "ml" / "data" / "raw" / "upi_transactions_2024.csv"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "upi_transactions_clean.csv"
)

EXPECTED_COLUMNS = [
    "transaction id",
    "timestamp",
    "transaction type",
    "merchant_category",
    "amount (INR)",
    "transaction_status",
    "sender_age_group",
    "receiver_age_group",
    "sender_state",
    "sender_bank",
    "receiver_bank",
    "device_type",
    "network_type",
    "fraud_flag",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

COLUMN_RENAMES = {
    "transaction id": "transaction_id",
    "transaction type": "transaction_type",
    "amount (INR)": "amount_inr",
}

STRING_COLUMNS = [
    "transaction id",
    "transaction type",
    "merchant_category",
    "transaction_status",
    "sender_age_group",
    "receiver_age_group",
    "sender_state",
    "sender_bank",
    "receiver_bank",
    "device_type",
    "network_type",
    "day_of_week",
]

VALID_STATUSES = {"SUCCESS", "FAILED"}


def validate_schema(dataframe: pd.DataFrame) -> None:
    actual_columns = dataframe.columns.tolist()
    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual: {actual_columns}"
        )


def clean_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    validate_schema(dataframe)
    if dataframe.isna().any().any():
        missing = dataframe.isna().sum()
        raise ValueError(f"Missing values found:\n{missing[missing.gt(0)]}")

    cleaned = dataframe.copy()
    rows_before = len(cleaned)
    cleaned = cleaned.drop_duplicates().copy()
    duplicates_removed = rows_before - len(cleaned)

    for column in STRING_COLUMNS:
        cleaned[column] = cleaned[column].astype("string").str.strip()
        if cleaned[column].eq("").any():
            raise ValueError(f"Blank values found in {column!r}")

    cleaned["transaction_status"] = cleaned["transaction_status"].str.upper()
    invalid_status = ~cleaned["transaction_status"].isin(VALID_STATUSES)
    if invalid_status.any():
        values = cleaned.loc[invalid_status, "transaction_status"].unique().tolist()
        raise ValueError(f"Invalid transaction statuses: {values}")

    cleaned["amount (INR)"] = pd.to_numeric(
        cleaned["amount (INR)"],
        errors="coerce",
    )
    if cleaned["amount (INR)"].isna().any():
        raise ValueError("Invalid transaction amounts found")
    if cleaned["amount (INR)"].le(0).any():
        raise ValueError("Non-positive transaction amount found")
    if cleaned["amount (INR)"].mod(1).ne(0).any():
        raise ValueError("Non-integral transaction amount found")
    cleaned["amount (INR)"] = cleaned["amount (INR)"].astype("int64")

    cleaned["fraud_flag"] = pd.to_numeric(
        cleaned["fraud_flag"],
        errors="coerce",
    )
    if cleaned["fraud_flag"].isna().any():
        raise ValueError("Invalid fraud flags found")
    invalid_fraud = ~cleaned["fraud_flag"].isin({0, 1})
    if invalid_fraud.any():
        values = cleaned.loc[invalid_fraud, "fraud_flag"].unique().tolist()
        raise ValueError(f"Invalid fraud flags: {values}")
    cleaned["fraud_flag"] = cleaned["fraud_flag"].astype("int8")

    cleaned["timestamp"] = pd.to_datetime(
        cleaned["timestamp"],
        errors="coerce",
    )
    if cleaned["timestamp"].isna().any():
        raise ValueError("Invalid timestamps found")

    expected_hour = cleaned["timestamp"].dt.hour.astype("int8")
    expected_day_name = cleaned["timestamp"].dt.day_name()
    expected_weekend = cleaned["timestamp"].dt.dayofweek.ge(5).astype("int8")

    supplied_hour = pd.to_numeric(cleaned["hour_of_day"], errors="coerce")
    supplied_weekend = pd.to_numeric(cleaned["is_weekend"], errors="coerce")
    hour_mismatches = supplied_hour.ne(expected_hour).sum()
    day_mismatches = (
        cleaned["day_of_week"].str.casefold().ne(expected_day_name.str.casefold()).sum()
    )
    weekend_mismatches = supplied_weekend.ne(expected_weekend).sum()
    if hour_mismatches or day_mismatches or weekend_mismatches:
        raise ValueError(
            "Timestamp-derived columns are inconsistent: "
            f"hour={hour_mismatches}, day={day_mismatches}, "
            f"weekend={weekend_mismatches}"
        )

    # Recompute these fields so the processed values always come from timestamp.
    cleaned["hour_of_day"] = expected_hour
    cleaned["day_of_week"] = expected_day_name
    cleaned["is_weekend"] = expected_weekend

    cleaned = cleaned.rename(columns=COLUMN_RENAMES)
    if cleaned["transaction_id"].duplicated().any():
        duplicate_ids = cleaned.loc[
            cleaned["transaction_id"].duplicated(keep=False),
            "transaction_id",
        ].unique()
        raise ValueError(f"Duplicate transaction IDs found: {duplicate_ids[:10].tolist()}")

    return cleaned, duplicates_removed


def print_summary(dataframe: pd.DataFrame, duplicates_removed: int) -> None:
    print("\n=== CLEANING SUMMARY ===")
    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {len(dataframe.columns)}")
    print(f"Missing: {int(dataframe.isna().sum().sum())}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Duplicate transaction IDs: {int(dataframe['transaction_id'].duplicated().sum())}")

    print("\nStatus:")
    print(dataframe["transaction_status"].value_counts())

    print("\nAmount:")
    print(dataframe["amount_inr"].describe())


def run(input_path: Path, output_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(input_path)
    cleaned, duplicates_removed = clean_dataframe(raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")

    print(f"Saved: {output_path}")
    print_summary(cleaned, duplicates_removed)
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and deterministically clean the Kaggle UPI dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.input, arguments.output)
