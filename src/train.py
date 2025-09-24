
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from src.data_loader import CustomDataGenerator
from src.models import CustomModel
import argparse
from config.config import EPOCHS, LEARNING_RATE, TRAIN_DIR, VAL_DIR, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, SAVED_MODEL_DIR , CLASS_NAMES, BACKBONE
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
import os

def train(backbone=BACKBONE, learning_rate=LEARNING_RATE, epochs=EPOCHS, batch_size=BATCH_SIZE):
    print(f"[INFO] Training with backbone: {backbone}")
    print(f"[INFO] Learning rate: {learning_rate}, Epochs: {epochs}")

    train_ds = CustomDataGenerator(
        directory=TRAIN_DIR, 
        classes=CLASS_NAMES,
        batch_size=batch_size, 
        img_size=(IMG_HEIGHT, IMG_WIDTH), 
        shuffle=True,
        augmentation=True
    )
    
    val_ds = CustomDataGenerator(
        directory=VAL_DIR, 
        classes=CLASS_NAMES, 
        batch_size=batch_size, 
        img_size=(IMG_HEIGHT, IMG_WIDTH), 
        shuffle=False,
        augmentation=False
    )

    # build model
    model = CustomModel(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        num_classes=NUM_CLASSES,
        backbone=backbone
    ).model 

    optimizer = Adam(learning_rate=learning_rate)
    loss = CategoricalCrossentropy()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        CSVLogger("logs/training_log.csv")
    ]

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # ensure save dir exists
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    save_path = os.path.join(SAVED_MODEL_DIR, f"{backbone}_model.keras")
    model.save(save_path)
    print(f"[INFO] Model saved at {save_path}")

    return history


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", 
        type=str, 
        default=BACKBONE, 
        choices=["customCNN", "inception_v3", "mobilenet_v2", "ViT"])
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=LEARNING_RATE)
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=EPOCHS
        )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=BATCH_SIZE, 
        help="Batch size for training"
    )
    args = parser.parse_args()

    train(backbone=args.model, learning_rate=args.learning_rate, epochs=args.epochs, batch_size=args.batch_size)