import numpy as np
import os
import argparse
from config.config import CLASS_NAMES, BACKBONE, SAVED_MODEL_DIR, TEST_DIR
from src.utils import preprocess_image_from_path
from keras.models import load_model


def load_backbone_model(backbone=BACKBONE):
    model_path = os.path.join(SAVED_MODEL_DIR, f"{backbone}_model.keras")
    print(f"Loading model: {backbone}")
    return load_model(model_path)


def predict_image(image_path, backbone=BACKBONE):
    """Inference trên 1 ảnh"""
    model = load_backbone_model(backbone)
    img_array = preprocess_image_from_path(image_path)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    idx = np.argmax(predictions[0])
    confidence = predictions[0][idx]

    label = CLASS_NAMES[idx]
    conf = float(confidence)

    print(f"Ảnh: {image_path} -> Dự đoán: {label} ({conf*100:.2f}%)")
    return label, conf


def predict_images(folder_path, backbone=BACKBONE):
    """Inference trên nhiều ảnh trong folder và subfolder"""
    model = load_backbone_model(backbone)

    image_files = []
    file_labels = []

    # Duyệt qua tất cả subfolder
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, f)
                image_files.append(img_path)
                # Lấy tên subfolder (vd: banana/, apple/, ...)
                label_name = os.path.basename(os.path.dirname(img_path))
                file_labels.append(label_name)

    if not image_files:
        print("Không có ảnh hợp lệ trong folder.")
        return

    images, valid_files, valid_labels = [], [], []
    for img_path, lbl in zip(image_files, file_labels):
        try:
            img_array = preprocess_image_from_path(img_path)
            images.append(img_array)
            valid_files.append(img_path)
            valid_labels.append(lbl)
        except Exception as e:
            print(f"Lỗi với ảnh {img_path}: {e}")

    if not images:
        print("Không có ảnh nào được load thành công.")
        return

    images = np.array(images)
    predictions = model.predict(images, verbose=0)

    for i, pred in enumerate(predictions):
        idx = np.argmax(pred)
        confidence = pred[idx]
        print(f"Ảnh: {valid_files[i]} "
              f"Label thực tế: {valid_labels[i]} "
              f"Dự đoán: {CLASS_NAMES[idx]} ({confidence*100:.2f}%)")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference ảnh với model đã huấn luyện")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn ảnh để dự đoán")
    parser.add_argument("--folder", type=str, default=None, help="Đường dẫn folder chứa ảnh để dự đoán")
    parser.add_argument("--model", type=str, default=BACKBONE,  choices=["customCNN", "inception_v3", "mobilenet_v2", "ViT"])

    args = parser.parse_args()

    if args.image:
        predict_image(args.image, backbone=args.model)
    elif args.folder:
        predict_images(args.folder, backbone=args.model)
    else:
        print("Vui lòng cung cấp --image hoặc --folder để chạy inference.")