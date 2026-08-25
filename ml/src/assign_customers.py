from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "upi_transactions_clean.csv"
)
DEFAULT_CUSTOMERS_PATH = REPO_ROOT / "ml" / "data" / "processed" / "customers.csv"
DEFAULT_TRANSACTIONS_PATH = (
    REPO_ROOT
    / "ml"
    / "data"
    / "processed"
    / "transactions_with_customers.csv"
)
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT
    / "ml"
    / "data"
    / "processed"
    / "customer_assignment_summary.json"
)

STABLE_COLUMNS = ["sender_age_group", "sender_state", "sender_bank"]
EXPECTED_COLUMNS = [
    "transaction_id",
    "timestamp",
    "transaction_type",
    "merchant_category",
    "amount_inr",
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

ARCHETYPES = np.array(
    [
        "infrequent",
        "regular",
        "heavy",
        "high_value",
        "unstable",
        "weekend_heavy",
        "night_heavy",
    ]
)
ARCHETYPE_PROBABILITIES = np.array([0.24, 0.56, 0.08, 0.05, 0.04, 0.02, 0.01])
ACTIVITY_MULTIPLIERS = {
    "infrequent": 0.35,
    "regular": 1.0,
    "heavy": 4.0,
    "high_value": 1.35,
    "unstable": 1.15,
    "weekend_heavy": 1.2,
    "night_heavy": 1.2,
}


@dataclass(frozen=True)
class AssignmentConfig:
    customer_count: int = 10_000
    minimum_transactions: int = 5
    seed: int = 42


def _validate_input(dataframe: pd.DataFrame) -> None:
    if dataframe.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected cleaned schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual: {dataframe.columns.tolist()}"
        )
    if dataframe["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found")
    if dataframe.isna().any().any():
        raise ValueError("Missing values found in cleaned transactions")


def _allocate_customers(
    stratum_counts: pd.Series,
    config: AssignmentConfig,
) -> pd.Series:
    capacities = (stratum_counts // config.minimum_transactions).clip(lower=1)
    if int(capacities.sum()) < config.customer_count:
        raise ValueError(
            "Requested customer count cannot satisfy the minimum transaction constraint"
        )

    quotas = stratum_counts / stratum_counts.sum() * config.customer_count
    allocation = np.floor(quotas).astype(int).clip(lower=1)
    allocation = np.minimum(allocation, capacities).astype(int)

    while int(allocation.sum()) < config.customer_count:
        candidates = allocation.lt(capacities)
        if not candidates.any():
            raise ValueError("No stratum capacity remains for customer allocation")
        priority = (quotas - allocation).where(candidates, -np.inf)
        allocation.loc[priority.idxmax()] += 1

    while int(allocation.sum()) > config.customer_count:
        candidates = allocation.gt(1)
        priority = (allocation - quotas).where(candidates, -np.inf)
        allocation.loc[priority.idxmax()] -= 1

    return allocation.astype(int)


def _sample_preference(
    values: pd.Series,
    customer_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = values.value_counts().sort_index()
    return rng.choice(
        counts.index.to_numpy(),
        size=customer_count,
        p=(counts / counts.sum()).to_numpy(),
    )


def _mode(series: pd.Series) -> str:
    counts = series.value_counts()
    maximum = counts.max()
    return str(sorted(counts[counts.eq(maximum)].index.astype(str))[0])


def _assign_stratum(
    transactions: pd.DataFrame,
    customer_ids: np.ndarray,
    archetypes: np.ndarray,
    preferred_devices: np.ndarray,
    preferred_networks: np.ndarray,
    rng: np.random.Generator,
    minimum_transactions: int,
    high_amount_threshold: float,
) -> np.ndarray:
    customer_count = len(customer_ids)
    row_count = len(transactions)
    if row_count < customer_count * minimum_transactions:
        raise ValueError("Stratum cannot satisfy minimum transactions per customer")

    assignments = np.empty(row_count, dtype=object)
    assigned_counts = np.zeros(customer_count, dtype=np.int64)
    cursor = 0

    # Guarantee that every generated customer has a usable history.
    for _ in range(minimum_transactions):
        order = rng.permutation(customer_count)
        assignments[cursor : cursor + customer_count] = customer_ids[order]
        assigned_counts[order] += 1
        cursor += customer_count

    activity = np.array(
        [ACTIVITY_MULTIPLIERS[str(archetype)] for archetype in archetypes],
        dtype=float,
    )
    activity *= rng.lognormal(mean=0.0, sigma=0.35, size=customer_count)

    for position in range(cursor, row_count):
        row = transactions.iloc[position]
        scores = activity.copy()
        scores *= np.where(preferred_devices == row["device_type"], 2.2, 0.72)
        scores *= np.where(preferred_networks == row["network_type"], 1.8, 0.78)

        if row["transaction_status"] == "FAILED":
            scores *= np.where(archetypes == "unstable", 3.5, 1.0)
        else:
            scores *= np.where(archetypes == "unstable", 0.82, 1.0)

        if float(row["amount_inr"]) >= high_amount_threshold:
            scores *= np.where(archetypes == "high_value", 3.0, 1.0)
        else:
            scores *= np.where(archetypes == "high_value", 0.72, 1.0)

        if int(row["is_weekend"]) == 1:
            scores *= np.where(archetypes == "weekend_heavy", 3.5, 1.0)
        else:
            scores *= np.where(archetypes == "weekend_heavy", 0.72, 1.0)

        is_night = int(row["hour_of_day"]) >= 20 or int(row["hour_of_day"]) <= 5
        if is_night:
            scores *= np.where(archetypes == "night_heavy", 3.5, 1.0)
        else:
            scores *= np.where(archetypes == "night_heavy", 0.72, 1.0)

        # Keep activity heavy-tailed without allowing one customer to absorb a stratum.
        scores /= np.sqrt(np.maximum(assigned_counts, 1) / minimum_transactions)
        probabilities = scores / scores.sum()
        customer_index = int(rng.choice(customer_count, p=probabilities))
        assignments[position] = customer_ids[customer_index]
        assigned_counts[customer_index] += 1

    return assignments


def assign_customers(
    dataframe: pd.DataFrame,
    config: AssignmentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_input(dataframe)
    transactions = dataframe.copy()
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], errors="raise")
    transactions = transactions.sort_values(
        ["timestamp", "transaction_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = transactions.groupby(STABLE_COLUMNS, sort=True, observed=True)
    stratum_counts = grouped.size()
    allocation = _allocate_customers(stratum_counts, config)
    high_amount_threshold = float(transactions["amount_inr"].quantile(0.75))
    rng = np.random.default_rng(config.seed)

    assigned_frames: list[pd.DataFrame] = []
    profile_seeds: list[dict[str, Any]] = []
    next_customer_number = 1

    for stratum, stratum_transactions in grouped:
        stratum_transactions = stratum_transactions.sort_values(
            ["timestamp", "transaction_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        stratum_customer_count = int(allocation.loc[stratum])
        customer_ids = np.array(
            [
                f"C{customer_number:06d}"
                for customer_number in range(
                    next_customer_number,
                    next_customer_number + stratum_customer_count,
                )
            ]
        )
        next_customer_number += stratum_customer_count

        archetypes = rng.choice(
            ARCHETYPES,
            size=stratum_customer_count,
            p=ARCHETYPE_PROBABILITIES,
        )
        preferred_devices = _sample_preference(
            stratum_transactions["device_type"],
            stratum_customer_count,
            rng,
        )
        preferred_networks = _sample_preference(
            stratum_transactions["network_type"],
            stratum_customer_count,
            rng,
        )
        assignments = _assign_stratum(
            stratum_transactions,
            customer_ids,
            archetypes,
            preferred_devices,
            preferred_networks,
            rng,
            config.minimum_transactions,
            high_amount_threshold,
        )
        stratum_transactions.insert(1, "customer_id", assignments)
        assigned_frames.append(stratum_transactions)

        for index, customer_id in enumerate(customer_ids):
            profile_seeds.append(
                {
                    "customer_id": customer_id,
                    "age_group": stratum[0],
                    "state": stratum[1],
                    "bank": stratum[2],
                    "synthetic_archetype": str(archetypes[index]),
                    "preferred_device": str(preferred_devices[index]),
                    "preferred_network": str(preferred_networks[index]),
                }
            )

    assigned = pd.concat(assigned_frames, ignore_index=True)
    assigned = assigned.sort_values(
        ["customer_id", "timestamp", "transaction_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    aggregates = (
        assigned.groupby("customer_id", sort=True, observed=True)
        .agg(
            transaction_count=("transaction_id", "size"),
            first_transaction_at=("timestamp", "min"),
            last_transaction_at=("timestamp", "max"),
            primary_device=("device_type", _mode),
            primary_network=("network_type", _mode),
        )
        .reset_index()
    )
    profiles = pd.DataFrame(profile_seeds).merge(
        aggregates,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )
    profiles = profiles.sort_values("customer_id").reset_index(drop=True)
    return profiles, assigned


def validate_assignment(
    source: pd.DataFrame,
    customers: pd.DataFrame,
    assigned: pd.DataFrame,
    config: AssignmentConfig,
) -> dict[str, Any]:
    if len(assigned) != len(source):
        raise ValueError("Assignment changed the transaction count")
    if assigned["customer_id"].isna().any():
        raise ValueError("Transactions without customers found")
    if assigned["transaction_id"].duplicated().any():
        raise ValueError("A transaction was assigned more than once")
    if assigned["transaction_id"].nunique() != source["transaction_id"].nunique():
        raise ValueError("Assignment lost source transactions")
    if len(customers) != config.customer_count:
        raise ValueError("Unexpected synthetic customer count")
    if customers["transaction_count"].lt(config.minimum_transactions).any():
        raise ValueError("Customer below minimum transaction count found")

    profile_lookup = customers.set_index("customer_id")
    for transaction_column, profile_column in (
        ("sender_age_group", "age_group"),
        ("sender_state", "state"),
        ("sender_bank", "bank"),
    ):
        expected = assigned["customer_id"].map(profile_lookup[profile_column])
        if not expected.eq(assigned[transaction_column]).all():
            raise ValueError(f"Unstable customer profile field: {profile_column}")

    chronological = assigned.groupby("customer_id", sort=False)["timestamp"].apply(
        lambda values: values.is_monotonic_increasing
    )
    if not chronological.all():
        raise ValueError("Non-chronological customer history found")

    counts = customers["transaction_count"]
    device_consistency = (
        assigned["device_type"]
        .eq(assigned["customer_id"].map(profile_lookup["primary_device"]))
        .mean()
    )
    network_consistency = (
        assigned["network_type"]
        .eq(assigned["customer_id"].map(profile_lookup["primary_network"]))
        .mean()
    )
    return {
        "seed": config.seed,
        "customers": len(customers),
        "transactions": len(assigned),
        "minimum_transactions": int(counts.min()),
        "median_transactions": float(counts.median()),
        "mean_transactions": float(counts.mean()),
        "p95_transactions": float(counts.quantile(0.95)),
        "maximum_transactions": int(counts.max()),
        "stable_profile_mismatches": 0,
        "chronological_histories": True,
        "device_primary_consistency": round(float(device_consistency), 4),
        "network_primary_consistency": round(float(network_consistency), 4),
        "archetype_counts": {
            str(key): int(value)
            for key, value in customers["synthetic_archetype"].value_counts().items()
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    input_path: Path,
    customers_path: Path,
    transactions_path: Path,
    summary_path: Path,
    config: AssignmentConfig,
) -> dict[str, Any]:
    source = pd.read_csv(input_path, parse_dates=["timestamp"])
    customers, assigned = assign_customers(source, config)
    summary = validate_assignment(source, customers, assigned, config)

    customers_path.parent.mkdir(parents=True, exist_ok=True)
    customers.to_csv(
        customers_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    assigned.to_csv(
        transactions_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    summary["customers_sha256"] = _sha256(customers_path)
    summary["transactions_sha256"] = _sha256(transactions_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=== SYNTHETIC CUSTOMER ASSIGNMENT ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved customers: {customers_path}")
    print(f"Saved transactions: {transactions_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign cleaned UPI transactions to synthetic customer histories."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--customers-output", type=Path, default=DEFAULT_CUSTOMERS_PATH)
    parser.add_argument(
        "--transactions-output",
        type=Path,
        default=DEFAULT_TRANSACTIONS_PATH,
    )
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--customers", type=int, default=10_000)
    parser.add_argument("--minimum-transactions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        input_path=arguments.input,
        customers_path=arguments.customers_output,
        transactions_path=arguments.transactions_output,
        summary_path=arguments.summary_output,
        config=AssignmentConfig(
            customer_count=arguments.customers,
            minimum_transactions=arguments.minimum_transactions,
            seed=arguments.seed,
        ),
    )
