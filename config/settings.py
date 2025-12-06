import os
from dotenv import load_dotenv

load_dotenv()

IMAGES_FOLDER = os.getenv("IMAGES_FOLDER", "./data/images")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
LOGS_PATH = os.getenv("LOGS_PATH", "./data/logs")

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")

CLIP_MODEL = os.getenv("CLIP_MODEL", "ViT-B/32")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

IMAGE_WEIGHT = float(os.getenv("IMAGE_WEIGHT", "0.7"))
TEXT_WEIGHT = float(os.getenv("TEXT_WEIGHT", "0.3"))

TOP_K_RESULTS = 10

os.makedirs(IMAGES_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
