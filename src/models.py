import tensorflow as tf
from tensorflow.keras import layers, Model
from config.config import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, BACKBONE
from tensorflow.keras.applications import InceptionV3, MobileNetV2
from keras_hub.models import ViTBackbone

class CustomModel:
    def __init__(self, input_shape=(IMG_HEIGHT,IMG_WIDTH,3), num_classes=NUM_CLASSES,backbone=BACKBONE):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.backbone = backbone
        self.model = self._build_model()

    def _build_model(self):
        if self.backbone == "customCNN":
            return self._build_custom_cnn()
        elif self.backbone == "inception_v3":
            return self._finetune_inception()
        elif self.backbone == "mobilenet_v2":
            return self._finetune_mobilenet()
        elif self.backbone == "ViT":
            return self._finetune_vit()
        else:
            raise ValueError(f"backbone {self.backbone} is not supported")

    def conv_block(self, x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        return x

    def residual_block(self, x, filters, strides=(1, 1)):
        shortcut = x
        
        x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        if strides != (1, 1) or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.25)(x)
        
        return x
        
    def _build_custom_cnn(self):
        inputs = layers.Input(shape=self.input_shape)

        x = self.conv_block(inputs, 32, kernel_size=(3, 3), padding='same', activation='relu')
        x = self.residual_block(x, 32)
        
        x = self.conv_block(x, 64, kernel_size=(3, 3), padding='same', activation='relu')
        x = self.residual_block(x, 64)
        
        x = self.conv_block(x, 128, kernel_size=(3, 3), padding='same', activation='relu')
        x = self.residual_block(x, 128)
        
        x = self.conv_block(x, 256, kernel_size=(3, 3), padding='same', activation='relu')
        x = self.residual_block(x, 256)


        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return Model(inputs, outputs, name="CustomCNN_ResNetStyle")

    def _finetune_inception(self):
        base = InceptionV3(weights="imagenet", include_top=False, input_shape=self.input_shape)
        for layer in base.layers:
            layer.trainable = False

        x = base.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256,activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        return Model(inputs=base.input, outputs=outputs, name="InceptionV3_TL")
        
    def _finetune_mobilenet(self):
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=self.input_shape)
        
        for layer in base.layers:
            layer.trainable = False

        x = base.output
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(256,activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        
        return Model(inputs=base.input, outputs=outputs, name="MobileNetV2_TL")


    def _finetune_vit(self):
        backbone = ViTBackbone.from_preset("vit_base_patch16_224_imagenet")

        for layer in backbone.layers[:-10]:   # giữ nguyên trừ 10 layer cuối
            layer.trainable = False
        for layer in backbone.layers[-10:]:   # fine-tune các layer cuối
            layer.trainable = True

        inputs = layers.Input(shape=self.input_shape)
        x = backbone(inputs)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        return Model(inputs, outputs, name="Vision_transformer")

    
