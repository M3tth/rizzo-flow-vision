"""Thread-safe long-lived model service with deterministic postprocessing."""

import time
from threading import Lock

from .decisions import decode
from .prompts import compile_request
from .responses import Response
from .schema import Request


class Engine:
    def __init__(self, backend, max_tokens=8192, calibration=None):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.backend = backend
        self.max_tokens = max_tokens
        self.calibration = calibration
        if calibration and calibration.fingerprint != backend.metadata["fingerprint"]:
            raise ValueError(
                "Calibration was fitted for a different model/runtime/prompt configuration"
            )
        self._lock = Lock()

    def decide(self, request: Request | dict) -> dict:
        # Re-validate a serialized snapshot, also protecting mutable Pydantic objects.
        request = Request.model_validate(
            request.model_dump() if isinstance(request, Request) else request
        )
        started = time.perf_counter()
        with self._lock:
            acquired = time.perf_counter()
            prefix, jobs = compile_request(self.backend.tokenizer, request, self.max_tokens)
            encoded = time.perf_counter()
            logits, timing = self.backend.score(prefix, jobs, request.mode)
            answers = {}
            for job in jobs:
                question = request.questions[job.id]
                temperature = (
                    self.calibration.temperatures.get(question.type, 1.0)
                    if self.calibration
                    else 1.0
                )
                answers[job.id] = decode(question, logits[job.id], temperature)
                answers[job.id]["prompt_sha256"] = job.prompt_sha256
                answers[job.id]["input_tokens"] = len(job.tokens)
        response = {
            "model": self.backend.metadata,
            "mode": request.mode,
            "answers": answers,
            "calibration": self.calibration.model_dump() if self.calibration else None,
            "timing": {
                **timing,
                "queue_seconds": acquired - started,
                "compile_seconds": encoded - acquired,
                "total_seconds": time.perf_counter() - started,
            },
        }
        return Response.model_validate(response).model_dump()
