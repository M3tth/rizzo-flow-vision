"""Local HTTP API. One resident model, serialized GPU access, no external calls."""

from fastapi import FastAPI, HTTPException

from .schema import Request


def create_app(engine):
    app = FastAPI(
        title="Rizzo Flow",
        version="0.1.0",
        description="Typed decisions with local Spark-X2.5-4B; no text generation.",
    )

    @app.get("/health")
    def health():
        return {"status": "ready", "model": engine.backend.metadata}

    @app.post("/v1/decisions")
    def decisions(request: Request):
        try:
            return engine.decide(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app
