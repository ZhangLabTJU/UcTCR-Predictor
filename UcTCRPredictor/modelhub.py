"""
Download helpers – lazily fetch model weights from Hugging Face
and cache them under  UcTCRPredictor/species/<sp>/models/.
"""
from importlib import resources
from pathlib import Path
from huggingface_hub import hf_hub_download


_HF_REPO = "howie825/TCR-V-BERT"
_HF_FILES = {
    "human": "human/pytorch_model.bin",
    "mouse": "mouse/pytorch_model.bin",     # 改成实际文件名
}

def ensure_model(species: str) -> Path:
    """Return local .bin path, downloading from HF if missing."""
    model_dir = Path(resources.files(f"UcTCRPredictor.species.{species}.models"))
    model_path = model_dir / Path(_HF_FILES[species]).name
    if model_path.exists():
        return model_path

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[UcTCR] downloading {_HF_FILES[species]} …")
    hf_hub_download(
        repo_id=_HF_REPO,
        filename=_HF_FILES[species],
        local_dir=model_dir,
        local_dir_use_symlinks=False
    )
    print(f"[UcTCR] saved → {model_path}")
    return model_path
