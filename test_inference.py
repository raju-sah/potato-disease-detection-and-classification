from PIL import Image
import tensorflow as tf
import numpy as np
import os
import glob

# Try loading interpreter
interpreter = tf.lite.Interpreter(model_path="model/potato_quantized.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()

CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight"]
IMG_SIZE = 256

def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

files = sorted(glob.glob("static/samples/*.jpg"))
for f in files:
    img = Image.open(f)
    arr = preprocess(img)
    interpreter.set_tensor(inp[0]["index"], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(out[0]["index"])[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    print(f"{os.path.basename(f)} -> {pred_class} (Confidence: {probs[pred_idx]*100:.2f}%)")
    print("   Raw probs:", probs)
