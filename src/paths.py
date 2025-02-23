from pathlib import Path


PROJECT_PATH = Path(__file__).resolve().parent.parent

TMP_IMAGES_DIR = PROJECT_PATH / 'tmp_images'
TMP_IMAGES_DIR.mkdir(exist_ok=True)

CHECKPOINT_DIR = PROJECT_PATH / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)