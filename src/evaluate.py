import os
import tensorflow as tf 
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from src.data_loader import CustomDataGenerator   
from config.config import CLASS_NAMES, BATCH_SIZE, IMG_SIZE, TEST_DIR, OUTPUT_DIR, BACKBONE, SAVED_MODEL_DIR
from keras.models import load_model


def evaluate_model(backbone=BACKBONE, test_folder=TEST_DIR, output_folder=OUTPUT_DIR):

    os.makedirs(output_folder, exist_ok=True)

    model_path = os.path.join(SAVED_MODEL_DIR, f"{backbone}_model.keras")
    print(f"Evaluating model: {model_path}")

    model = load_model(model_path)

    test_gen = CustomDataGenerator(directory=test_folder,
                                   classes=CLASS_NAMES,
                                   batch_size=BATCH_SIZE,
                                   img_size=IMG_SIZE,
                                   shuffle=False,
                                   augmentation=False)


    preds = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    y_true = []
    for _, batch_y in test_gen:
        y_true.extend(np.argmax(batch_y, axis=1))
    y_true = np.array(y_true)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    # tên file csv kết quả theo tên model
    output_csv = os.path.join(output_folder, f"{backbone}_eval.csv")

    results = {
        "accuracy": [acc],
        "precision": [prec],
        "recall": [rec],
        "f1_score": [f1]
    }
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(f"kết quả model {backbone} lưu vào {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=BACKBONE,
        choices=["customCNN", "inception_v3", "mobilenet_v2", "ViT"]
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=TEST_DIR
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR
    )
    args = parser.parse_args()

    evaluate_model(
        backbone=args.model,
        test_folder=args.test_dir,
        output_folder=args.output_dir
    )
