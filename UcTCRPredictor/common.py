from importlib import resources
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("UcTCRPredictor")

def get_model_dir(species: str) -> Path:
    """ Return packaged resources directory (pkg_resources or importlib.resources)."""
    return Path(resources.files(f"UcTCRPredictor.species.{species}.models"))
