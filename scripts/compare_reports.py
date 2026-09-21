"""Compare distributions by semantic option ID across precision or perturbation reports."""

import argparse
import json
from pathlib import Path

from rizzo_flow.cli import write_json


def compare(reference, candidate, perturbations=False):
    original = {row["id"]: row["response"] for row in reference["rows"]}
    rows = []
    for row in candidate["rows"]:
        key = row["id"]
        if perturbations:
            key = key.removesuffix("-reordered").removesuffix("-irrelevant")
        if key not in original:
            raise ValueError(f"Missing reference for {row['id']}")
        for question_id, answer in row["response"]["answers"].items():
            other = original[key]["answers"][question_id]
            a, b = other["probabilities"], answer["probabilities"]
            if a.keys() != b.keys():
                raise ValueError(f"Candidate IDs changed for {key}/{question_id}")
            rows.append(
                {
                    "id": row["id"],
                    "question": question_id,
                    "changed_argmax": max(a, key=a.get) != max(b, key=b.get),
                    "changed_status": other["status"] != answer["status"],
                    "max_probability_delta": max(abs(a[k] - b[k]) for k in a),
                    "reference_probabilities": a,
                    "candidate_probabilities": b,
                }
            )
    return {
        "reference_dataset_sha256": reference["dataset_sha256"],
        "candidate_dataset_sha256": candidate["dataset_sha256"],
        "summary": {
            "decisions": len(rows),
            "changed_argmaxes": sum(r["changed_argmax"] for r in rows),
            "changed_statuses": sum(r["changed_status"] for r in rows),
            "max_probability_delta": max((r["max_probability_delta"] for r in rows), default=0),
        },
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--perturbations", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = compare(
        json.loads(args.reference.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
        args.perturbations,
    )
    write_json(report, args.output)
