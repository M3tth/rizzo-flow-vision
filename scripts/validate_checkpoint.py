"""Run real-weight checks and preserve raw evidence. Execute from the repository root."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from fastapi.testclient import TestClient

from rizzo_flow.api import create_app
from rizzo_flow.backend import SparkBackend, selected_logits
from rizzo_flow.cli import read_jsonl, write_json
from rizzo_flow.engine import Engine
from rizzo_flow.evaluation import evaluate
from rizzo_flow.prompts import compile_request
from rizzo_flow.schema import Request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Spark-X2.5-4B")
    parser.add_argument("--bits", type=int, choices=[4, 8])
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-long-state", action="store_true")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    backend = SparkBackend.load(args.model, bits=args.bits)
    engine = Engine(backend)
    request = Request.model_validate_json(Path("examples/ticket.json").read_text())
    _, jobs = compile_request(backend.tokenizer, request, 8192)
    job = jobs[0]
    cache = backend._prefill(job.tokens[:-1])
    hidden = backend.model.model(mx.array([job.tokens[-1:]]), cache=cache)[:, -1, :]
    optimized = selected_logits(backend.model, hidden, job.slots)
    head = backend.model.model.embedding
    native = head.as_linear(hidden)[:, mx.array(job.slots)].astype(mx.float32)
    mx.eval(optimized, native)
    projection_delta = mx.max(mx.abs(optimized - native)).item()
    if projection_delta > 0.125:
        raise RuntimeError(f"Selected projection diverged from full head: {projection_delta}")
    del cache, hidden, native, optimized
    print(f"Projection max logit delta: {projection_delta}", flush=True)
    with TestClient(create_app(engine)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/v1/decisions", json=request.model_dump())
        assert response.status_code == 200, response.text
        write_json(response.json(), out / "api-example.json")
    # State long enough to cross Spark's 512-token rotating attention window.
    long_request = request.model_dump()
    long_request["state"]["background"] = "Archived note: the office has blue walls. " * 150
    started = time.perf_counter()
    reports = {}
    suites = {
        "smoke": read_jsonl("benchmarks/smoke.jsonl"),
        "perturbations": read_jsonl("benchmarks/perturbations.jsonl"),
        "long-state": [{"id": "long-shared-state", "request": long_request}],
    }
    if args.skip_long_state:
        del suites["long-state"]
    for name, fixtures in suites.items():
        report = evaluate(
            engine, fixtures, repeats=2 if name == "long-state" else 1, compare_modes=True
        )
        write_json(report, out / f"{name}.json")
        reports[name] = report["summary"]
        print(name, json.dumps(report["summary"], ensure_ascii=False), flush=True)
    write_json(
        {
            "hardware": mx.device_info(),
            "model": backend.metadata,
            "selected_projection_max_logit_delta": projection_delta,
            "validation_seconds": time.perf_counter() - started,
            "suites": reports,
        },
        out / "summary.json",
    )


if __name__ == "__main__":
    main()
