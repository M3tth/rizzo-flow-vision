from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    size: str
    repo: str
    revision: str
    hidden_size: int  # identifies a checkpoint directory regardless of its path

    @property
    def path(self) -> Path:
        return Path("models") / self.repo.split("/")[1]


# Same Spark2.5 architecture, tokenizer and 1M-token context; pinned original weights.
MODELS = {
    spec.size: spec
    for spec in (
        ModelSpec("4b", "XHToken/Spark-X2.5-4B", "0bcb35678590218655dff3765b9e61c83b35e9c4", 2560),
        ModelSpec(
            "1.7b", "XHToken/Spark-X2.5-1.7B", "14d6e83c13c7add2b62a7c39b2131f4ed1cddcf8", 2048
        ),
    )
}
DEFAULT_SIZE = "4b"
MODEL_ID = MODELS[DEFAULT_SIZE].repo
MODEL_REVISION = MODELS[DEFAULT_SIZE].revision
RUNTIME_REVISION = "de2b4379fa1e2f2e1f99d84c83f0e008f651d86c"
DEFAULT_MODEL_PATH = MODELS[DEFAULT_SIZE].path


def identify(config: dict) -> ModelSpec:
    """Match a checkpoint's config.json to a supported, pinned model."""
    for spec in MODELS.values():
        if config.get("hidden_size") == spec.hidden_size:
            return spec
    raise ValueError(
        f"Unrecognized Spark2.5 checkpoint (hidden_size={config.get('hidden_size')}); "
        f"supported sizes: {', '.join(MODELS)}"
    )


def download_model(destination=None, size=DEFAULT_SIZE):
    from huggingface_hub import snapshot_download

    spec = MODELS[size]
    return snapshot_download(
        spec.repo,
        revision=spec.revision,
        local_dir=destination or spec.path,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.safetensors",
            "tokenizer.model",
            "LICENSE",
            "README.md",
        ],
    )
