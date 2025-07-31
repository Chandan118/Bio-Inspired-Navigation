from pathlib import Path
import torch


class Config:
    """Centralised experiment configuration."""

    # ----------------- PATHS ----------------- #
    ROOT_DIR = Path(__file__).resolve().parent.parent  # project root
    DATA_DIR = ROOT_DIR / "data" / "synthetic"
    TRAIN_IMG_DIR = DATA_DIR / "train" / "images"
    TRAIN_MASK_DIR = DATA_DIR / "train" / "masks"
    VAL_IMG_DIR = DATA_DIR / "val" / "images"
    VAL_MASK_DIR = DATA_DIR / "val" / "masks"

    MODELS_DIR = ROOT_DIR / "models" / "saved_models"
    OUTPUT_DIR = ROOT_DIR / "outputs" / "evaluation_results"

    # ----------------- DATA GENERATION ----------------- #
    TRAIN_SAMPLES = 200  # per modality
    VAL_SAMPLES = 50
    OCT_IMG_SIZE = (256, 256)
    LSCI_IMG_SIZE = (256, 256)

    # Noise params
    GAUSSIAN_NOISE_STD = 0.05
    MOTION_BLUR_KERNEL = 7

    # ----------------- TRAINING ----------------- #
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANDOM_SEED = 42

    BATCH_SIZE = 8
    NUM_EPOCHS = 10
    LR = 1e-3

    # ----------------- MODEL ----------------- #
    NUM_CLASSES = 1  # binary segmentation

    # --------------- MISC ------------------ #
    SAMPLE_PREDICTION_PATH = OUTPUT_DIR / "sample_prediction.png"
    REPORT_PATH = OUTPUT_DIR / "report.txt"