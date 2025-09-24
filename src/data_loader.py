import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential
from config.config import BATCH_SIZE, IMG_SIZE, CLASS_NAMES

class CustomDataGenerator(Sequence):
    def __init__(self, directory, classes = CLASS_NAMES, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=True, augmentation=False):
        self.directory = directory
        self.classes = classes  
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.augmentation = augmentation


        if self.augmentation:
            self.aug = Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.1),
            ])
        else:
            self.aug = None

        # tạo list (file_path, label)
        self.samples = []
        for label, class_name in enumerate(classes):
            class_dir = os.path.join(directory, class_name)
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath):
                    self.samples.append((fpath, label))

        self.on_epoch_end()
    
    def __len__(self):
        """số batch mỗi epoch"""
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        """trả về một batch (X, y)"""
        start = index*self.batch_size
        end = (index+1)*self.batch_size
        
        batch_samples = self.samples[start:end]
        images, labels = [], []

        for fpath, label in batch_samples:
            img = cv2.imread(fpath)                     
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            img = cv2.resize(img, self.img_size)        
            img = img.astype("float32") / 255.0         
            images.append(img)
            labels.append(label)

        X = np.array(images)
        y = np.array(labels)
        y = to_categorical(y, num_classes=len(self.classes))

        if self.aug is not None:
            X = tf.convert_to_tensor(X)
            X = self.aug(X, training=True)
            X = np.array(X)

        return X, y

    def on_epoch_end(self):
        """xao1 trộn sau mỗi epoch"""
        if self.shuffle:
            np.random.shuffle(self.samples)
