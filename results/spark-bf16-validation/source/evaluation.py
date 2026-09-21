"""Reproducible benchmark reports with coverage, proper scoring rules, and raw evidence."""

import copy
import hashlib
import math
import statistics

from .prompts import canonical


def evaluate(engine, fixtures: list[dict], repeats=1, compare_modes=False):
    if not fixtures or repeats < 1:
        raise ValueError("Provide fixtures and at least one repetition")
    # Explicit warmup is excluded from reported timings.
    engine.decide(fixtures[0]["request"])
    rows, latencies = [], []
    categorical, numeric, statuses, accepted = [], [], [], []
    mode_deltas, changed = [], 0
    for fixture in fixtures:
        responses = [engine.decide(fixture["request"]) for _ in range(repeats)]
        latencies.extend(r["timing"]["total_seconds"] for r in responses)
        response = responses[0]
        evidence = {
            "id": fixture["id"],
            "expected": fixture.get("expected", {}),
            "response": response,
            "repeat_timings": [r["timing"] for r in responses],
        }
        for key, expected in fixture.get("expected", {}).items():
            answer = response["answers"][key]
            is_accepted = answer["status"] == "ok"
            accepted.append(is_accepted)
            if "status" in expected:
                statuses.append(answer["status"] == expected["status"])
            if "label" in expected:
                ps = answer["probabilities"]
                label = expected["label"]
                if label not in ps:
                    raise ValueError(f"Unknown expected label {label} in fixture {fixture['id']}")
                predicted = max(ps, key=ps.get)
                categorical.append(
                    {
                        "correct": predicted == label,
                        "accepted": is_accepted,
                        "nll": -math.log(max(ps[label], 1e-300)),
                        "brier": sum((p - (k == label)) ** 2 for k, p in ps.items()),
                        "confidence": max(ps.values()),
                    }
                )
            if "value" in expected:
                target = expected["value"]
                if not isinstance(target, (int, float)) or not math.isfinite(target):
                    raise ValueError("Expected numeric targets must be finite")
                value = answer.get("value", answer.get("score"))
                numeric.append(None if value is None else float(value) - target)
        if compare_modes:
            alternate = copy.deepcopy(fixture["request"])
            alternate["mode"] = "direct" if response["mode"] == "shared" else "shared"
            other = engine.decide(alternate)
            evidence["alternate_mode_response"] = other
            for key, answer in response["answers"].items():
                a, b = answer["probabilities"], other["answers"][key]["probabilities"]
                mode_deltas.append(max(abs(a[k] - b[k]) for k in a))
                changed += max(a, key=a.get) != max(b, key=b.get)
        rows.append(evidence)

    def mean(values):
        return statistics.mean(values) if values else None

    bins, ece = [], 0.0
    for i in range(10):
        subset = [r for r in categorical if min(9, int(r["confidence"] * 10)) == i]
        if subset:
            accuracy = mean([r["correct"] for r in subset])
            confidence = mean([r["confidence"] for r in subset])
            ece += len(subset) / len(categorical) * abs(accuracy - confidence)
            bins.append(
                {
                    "lower": i / 10,
                    "count": len(subset),
                    "accuracy": accuracy,
                    "mean_top_probability": confidence,
                }
            )
    errors = [x for x in numeric if x is not None]
    sorted_times = sorted(latencies)
    summary = {
        "requests": len(fixtures),
        "repeats": repeats,
        "warmup_excluded": True,
        "latency_seconds": {
            "median": statistics.median(latencies),
            "p95": sorted_times[math.ceil(0.95 * len(sorted_times)) - 1],
        },
        "labeled_decision_coverage": mean(accepted),
        "status_accuracy": mean(statuses),
        "categorical": {
            "rows": len(categorical),
            "accuracy": mean([r["correct"] for r in categorical]),
            "accepted_accuracy": mean([r["correct"] for r in categorical if r["accepted"]]),
            "nll": mean([r["nll"] for r in categorical]),
            "brier": mean([r["brier"] for r in categorical]),
            "ece_10_bins": ece if categorical else None,
            "reliability_bins": bins,
        },
        "numeric": {
            "rows": len(numeric),
            "answered": len(errors),
            "mae_on_answered": mean([abs(x) for x in errors]),
            "rmse_on_answered": math.sqrt(mean([x * x for x in errors])) if errors else None,
        },
        "mode_comparison": {
            "decisions": len(mode_deltas),
            "changed_argmaxes": changed,
            "max_probability_delta": max(mode_deltas, default=0),
        }
        if compare_modes
        else None,
    }
    return {
        "dataset_sha256": hashlib.sha256(canonical(fixtures).encode()).hexdigest(),
        "summary": summary,
        "rows": rows,
    }
