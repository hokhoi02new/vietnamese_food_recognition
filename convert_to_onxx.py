import tensorflow as tf
import tf2onnx
import onnx


keras_model = tf.keras.models.load_model("saved_models/ViT_model.keras")

spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(keras_model,input_signature=spec)

onnx.save_model(onnx_model, "saved_models/ViT_model.onnx")
