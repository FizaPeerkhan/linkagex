
"""Evaluate LinkageX using real complaints from the dataset.

Run:
    python evaluate_linkage.py
    python evaluate_linkage.py --limit 50
    python evaluate_linkage.py --dataset "data/LINKAGEX_MASTER_CASES_V2.csv"

Keep the Flask backend running in another terminal.
Ground-truth labels are used only for evaluation, never sent to the API.
"""

import argparse
import math
import random
import statistics
import time
from pathlib import Path

import pandas as pd
import requests

LABEL_COLUMNS = [
    "incident_cluster_id",
    "campaign_id",
    "attacker_entity_id",
]
TOP_K = 5
RANDOM_SEED = 42


def clean(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if value.lower() in {
        "", "nan", "none", "null", "unknown", "n/a"
    }:
        return None

    return value


def get_labels(row):
    return {
        column: clean(row.get(column))
        for column in LABEL_COLUMNS
        if clean(row.get(column)) is not None
    }


def is_relevant(query_labels, candidate):
    candidate_labels = get_labels(candidate)

    return any(
        query_labels.get(column) is not None
        and query_labels.get(column) == candidate_labels.get(column)
        for column in LABEL_COLUMNS
    )


def analyze_complaint(session, base_url, complaint_text):
    response = session.post(
        f"{base_url}/api/analyze",
        json={"complaint_text": complaint_text},
        timeout=180,
    )
    response.raise_for_status()

    payload = response.json()

    if payload.get("status") != "success":
        raise RuntimeError(f"Analysis failed: {payload}")

    incident = payload.get("data", {})

    if isinstance(incident, dict):
        if isinstance(incident.get("incident"), dict):
            incident = incident["incident"]

    if not isinstance(incident, dict):
        raise RuntimeError(
            "The analyze endpoint did not return an incident object."
        )

    return incident


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LinkageX case retrieval."
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:5000",
    )
    parser.add_argument(
        "--dataset",
        default=str(
            Path(__file__).resolve().parent
            / "data"
            / "LINKAGEX_MASTER_CASES_V2.csv"
        ),
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--output",
        default="linkage_evaluation_results.csv",
    )

    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    dataset_path = Path(args.dataset).expanduser().resolve()

    if args.limit < 1:
        print("ERROR: --limit must be at least 1.")
        return 1

    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        print(
            'Run with --dataset "YOUR_ACTUAL_CSV_PATH" '
            "to specify the correct location."
        )
        return 1

    print(f"Dataset path: {dataset_path}")

    df = pd.read_csv(dataset_path, low_memory=False)
    print(f"Dataset rows: {len(df):,}")
    print(f"Dataset columns: {len(df.columns)}")

    required = {
        "case_id",
        "complaint_text",
        *LABEL_COLUMNS,
    }
    missing = required - set(df.columns)

    if missing:
        print(f"ERROR: Missing columns: {sorted(missing)}")
        return 1

    label_counts = {
        column: df[column].map(clean).value_counts().to_dict()
        for column in LABEL_COLUMNS
    }

    eligible_indices = []

    for idx, row in df.iterrows():
        complaint_text = clean(row.get("complaint_text"))
        labels = get_labels(row.to_dict())

        if not complaint_text or not labels:
            continue

        has_shared_label = any(
            label_counts[column].get(value, 0) > 1
            for column, value in labels.items()
        )

        if has_shared_label:
            eligible_indices.append(idx)

    if not eligible_indices:
        print(
            "ERROR: No complaints with text and shared ground-truth "
            "labels were found."
        )
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(eligible_indices)

    sample_indices = eligible_indices[:args.limit]

    print(f"Eligible complaints: {len(eligible_indices):,}")
    print(f"Selected test complaints: {len(sample_indices)}")
    print(
        "Relevance definition: at least one shared, non-empty "
        "cluster, campaign, or attacker label."
    )
    print("Ground-truth labels will NOT be sent to the API.\n")

    session = requests.Session()

    try:
        health = session.get(base_url + "/", timeout=10)
        health.raise_for_status()
    except requests.RequestException as exc:
        print(f"ERROR: Cannot reach Flask backend: {exc}")
        print("Start your Flask app and run this script again.")
        return 1

    output_rows = []
    errors = 0

    # Build label-to-case mappings once instead of scanning the full
    # dataset for every single query.
    label_to_case_ids = {column: {} for column in LABEL_COLUMNS}

    for _, row in df.iterrows():
        candidate_id = clean(row.get("case_id"))

        if not candidate_id:
            continue

        for column, value in get_labels(row.to_dict()).items():
            label_to_case_ids[column].setdefault(value, set()).add(
                candidate_id
            )

    for position, idx in enumerate(sample_indices, start=1):
        source = df.loc[idx].to_dict()
        case_id = clean(source.get("case_id")) or f"row_{idx}"
        complaint_text = clean(source.get("complaint_text"))
        query_labels = get_labels(source)

        started = time.perf_counter()

        try:
            # Run the real NLP extraction endpoint.
            incident = analyze_complaint(
                session, base_url, complaint_text
            )

            # Add only the source case ID for self-exclusion.
            # Never send cluster, campaign, or attacker labels.
            incident = dict(incident)
            incident["case_id"] = case_id
            incident["complaint_text"] = complaint_text

            response = session.post(
                base_url + "/api/linkage",
                json={
                    "incident": incident,
                    "complaint_text": complaint_text,
                },
                timeout=240,
            )
            response.raise_for_status()

            payload = response.json()

            if payload.get("status") != "success":
                raise RuntimeError(f"Linkage failed: {payload}")

            results = payload.get("results", []) or []

            # Remove the query case even if the backend returns it.
            results = [
                item for item in results
                if str(item.get("case_id", "")).strip() != case_id
            ][:TOP_K]

            relevant_ids = set()

            for column, value in query_labels.items():
                relevant_ids.update(
                    label_to_case_ids[column].get(value, set())
                )

            relevant_ids.discard(case_id)

            retrieved_ids = [
                str(item.get("case_id", "")).strip()
                for item in results
            ]

            relevant_retrieved = sum(
                retrieved_id in relevant_ids
                for retrieved_id in retrieved_ids
            )

            # Fixed-k Precision@5: missing result slots count as
            # non-relevant, as standard fixed-k evaluation requires.
            precision = relevant_retrieved / TOP_K

            recall = (
                relevant_retrieved / len(relevant_ids)
                if relevant_ids else math.nan
            )

            if math.isnan(recall):
                f1 = math.nan
            elif precision + recall == 0:
                f1 = 0.0
            else:
                f1 = (
                    2 * precision * recall
                    / (precision + recall)
                )

            reciprocal_rank = 0.0

            for rank, retrieved_id in enumerate(
                retrieved_ids, start=1
            ):
                if retrieved_id in relevant_ids:
                    reciprocal_rank = 1.0 / rank
                    break

            elapsed = time.perf_counter() - started

            output_rows.append({
                "query_case_id": case_id,
                "retrieved_count": len(results),
                "relevant_cases_in_dataset": len(relevant_ids),
                "relevant_retrieved_at_5": relevant_retrieved,
                "precision_at_5": precision,
                "recall_at_5": recall,
                "f1_at_5": f1,
                "reciprocal_rank": reciprocal_rank,
                "response_time_seconds": round(elapsed, 3),
                "retrieved_case_ids": ";".join(retrieved_ids),
                "error": "",
            })

            print(
                f"[{position}/{len(sample_indices)}] {case_id} | "
                f"Retrieved: {len(results)} | "
                f"Relevant@5: {relevant_retrieved} | "
                f"P@5: {precision:.3f} | "
                f"Time: {elapsed:.1f}s"
            )

        except Exception as exc:
            errors += 1
            elapsed = time.perf_counter() - started

            output_rows.append({
                "query_case_id": case_id,
                "retrieved_count": 0,
                "relevant_cases_in_dataset": 0,
                "relevant_retrieved_at_5": 0,
                "precision_at_5": math.nan,
                "recall_at_5": math.nan,
                "f1_at_5": math.nan,
                "reciprocal_rank": math.nan,
                "response_time_seconds": round(elapsed, 3),
                "retrieved_case_ids": "",
                "error": str(exc),
            })

            print(
                f"[{position}/{len(sample_indices)}] "
                f"{case_id} | ERROR: {exc}"
            )

    results_df = pd.DataFrame(output_rows)

    output_path = Path(args.output).expanduser().resolve()
    results_df.to_csv(output_path, index=False)

    successful = results_df[results_df["error"] == ""]

    print("\n" + "=" * 60)
    print("LINKAGEX EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Dataset rows: {len(df):,}")
    print(f"Eligible complaints: {len(eligible_indices):,}")
    print(f"Queries attempted: {len(sample_indices)}")
    print(f"Successful queries: {len(successful)}")
    print(f"Failed queries: {errors}")

    if not successful.empty:
        metrics = [
            ("precision_at_5", "Mean Precision@5"),
            ("recall_at_5", "Mean Recall@5"),
            ("f1_at_5", "Mean F1@5"),
            ("reciprocal_rank", "MRR"),
        ]

        for column, label in metrics:
            values = successful[column].dropna()

            if len(values):
                print(f"{label}: {values.mean():.4f}")
            else:
                print(f"{label}: N/A")

        times = successful["response_time_seconds"].dropna()

        if len(times):
            print(f"Mean response time: {times.mean():.2f}s")
            print(
                "Median response time: "
                f"{statistics.median(times):.2f}s"
            )

    print(f"\nDetailed results: {output_path}")
    print(
        "\nNOTE: Metrics measure agreement with the dataset's labels, "
        "not verified real-world investigative accuracy."
    )
    print(
        "Ensure the running backend uses the same CSV as the "
        "evaluation dataset."
    )

    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
