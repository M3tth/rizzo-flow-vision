import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_MODEL_PATH, download_model


def write_json(value, destination):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if destination:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Results are create-only; never silently overwrite benchmark evidence.
        with path.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Rizzo Flow — local Spark typed decisions")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download the pinned original Spark checkpoint")
    download.add_argument("--destination", type=Path, default=DEFAULT_MODEL_PATH)
    schema = commands.add_parser("schema", help="Print the JSON Schema for requests")
    schema.add_argument("--output")
    fit = commands.add_parser("calibrate", help="Fit temperatures on separate labeled logit rows")
    fit.add_argument("input", type=Path)
    fit.add_argument("--fingerprint", required=True)
    fit.add_argument("--output", required=True)
    for name in ("decide", "serve", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
        p.add_argument("--bits", type=int, choices=(4, 8))
        p.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--max-tokens", type=int, default=8192)
        p.add_argument("--calibration", type=Path)
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8017)
        else:
            p.add_argument("input", type=Path)
            p.add_argument("--output")
            if name == "evaluate":
                p.add_argument("--repeats", type=int, default=1)
                p.add_argument("--compare-modes", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "download":
            print(download_model(args.destination))
            return
        if args.command == "schema":
            from .schema import Request

            write_json(Request.model_json_schema(), args.output)
            return
        if args.command == "calibrate":
            from .calibration import fit_temperature

            write_json(
                fit_temperature(read_jsonl(args.input), args.fingerprint).model_dump(), args.output
            )
            return
        from .backend import SparkBackend
        from .calibration import Calibration
        from .engine import Engine

        # Validate the request before loading gigabytes of weights.
        if args.command == "decide":
            from .schema import Request

            request = Request.model_validate_json(args.input.read_text())
        backend = SparkBackend.load(
            args.model, bits=args.bits, device=args.device, batch_size=args.batch_size
        )
        calibration = Calibration.from_file(args.calibration) if args.calibration else None
        engine = Engine(backend, max_tokens=args.max_tokens, calibration=calibration)
        if args.command == "decide":
            write_json(engine.decide(request), args.output)
        elif args.command == "evaluate":
            from .evaluation import evaluate

            write_json(
                evaluate(engine, read_jsonl(args.input), args.repeats, args.compare_modes),
                args.output,
            )
        elif args.command == "serve":
            import uvicorn

            from .api import create_app

            uvicorn.run(create_app(engine), host=args.host, port=args.port)
    except (ValueError, OSError, ImportError) as error:
        print(f"rizzo: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
