from pathlib import Path

MODEL_ID = "XHToken/Spark-X2.5-4B"
MODEL_REVISION = "0bcb35678590218655dff3765b9e61c83b35e9c4"
RUNTIME_REVISION = "de2b4379fa1e2f2e1f99d84c83f0e008f651d86c"
DEFAULT_MODEL_PATH = Path("models/Spark-X2.5-4B")


def download_model(destination=DEFAULT_MODEL_PATH):
    from huggingface_hub import snapshot_download

    return snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=destination,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "*.safetensors",
            "tokenizer.model",
            "LICENSE",
            "README.md",
        ],
    )
