"""Central, environment-driven configuration shared by training, app and API."""
import os
from pathlib import Path

try:  # python-dotenv is optional (not installed in the slim serving image)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
PROJECT_NAME = os.getenv("PROJECT_NAME", "22l-6632_A2_Ops")
SEED = int(os.getenv("SEED", "42"))
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

# Label convention: 0 = real, 1 = AI-generated (the "positive" class is AI).
CLASS_NAMES = ["real", "ai"]

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "raw"))
SPLIT_DIR = Path(os.getenv("SPLIT_DIR", ROOT / "data" / "splits"))
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"

# Folder name (lower-case) -> class. The Kaggle dataset ships "AiArtData" and "RealArt".
FOLDER_TO_CLASS = {
    "aiartdata": "ai", "ai": "ai", "fake": "ai", "ai_generated": "ai",
    "realart": "real", "real": "real",
}
# Formats tf.io.decode_image can read (no WEBP/TIFF); used for splits and the web app.
TF_FORMATS = {"JPEG", "PNG", "BMP"}
