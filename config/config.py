import yaml

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# paths
DATA_DIR = config["paths"]["data_dir"]
TRAIN_DIR = config["paths"]["train_dir"]
VAL_DIR = config["paths"]["val_dir"]
TEST_DIR = config["paths"]["test_dir"]
SAVED_MODEL_DIR = config["paths"]["saved_model_dir"]
OUTPUT_DIR = config["paths"]["result_dir"]

# image
IMG_HEIGHT = config["image"]["height"]
IMG_WIDTH = config["image"]["width"]
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# classes
CLASS_NAMES = config["classes"]["names"]
NUM_CLASSES = len(CLASS_NAMES)

# training
EPOCHS = config["training"]["epochs"]
BATCH_SIZE = config["training"]["batch_size"]
LEARNING_RATE = config["training"]["learning_rate"]
BACKBONE = config["training"]["backbone"]
