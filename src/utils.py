import cv2
import numpy as np
from config.config import IMG_HEIGHT, IMG_WIDTH
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def preprocess_image_from_path(image_path):
    """chuẩn hóa ảnh (resize, normalize, add batch dimension)"""
    image = cv2.imread(image_path)            
    if image is None:
        raise ValueError(f"Không load được ảnh: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    img_array = image.astype("float32") / 255.0
    return img_array

def plot_history(history):
    """vẽ biểu đồ loss & accuracy"""
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12,5))


    plt.subplot(1,2,1)
    plt.plot(epochs, acc, "b-", label="Training acc")
    plt.plot(epochs, val_acc, "r-", label="Validation acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()


    plt.subplot(1,2,2)
    plt.plot(epochs, loss, "b-", label="Training loss")
    plt.plot(epochs, val_loss, "r-", label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.show()