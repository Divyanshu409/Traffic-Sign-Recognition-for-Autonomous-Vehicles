import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

model = keras.models.load_model("traffic_sign_recognition_model.keras")

with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

def predict_sign(image_path, top_k=3):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (32, 32))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_norm, axis=0)

    probs = model.predict(img_batch, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]

    print(f"\nImage: {image_path}")
    print(f"Prediction  : {CLASS_NAMES[top_indices[0]]}")
    print(f"Confidence  : {probs[top_indices[0]] * 100:.2f}%")
    print(f"\nTop {top_k} results:")
    for i, idx in enumerate(top_indices):
        print(f"  #{i+1}  {CLASS_NAMES[idx]:<40} {probs[idx] * 100:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    bar_labels = [CLASS_NAMES[i][:25] for i in top_indices]
    bar_values = [probs[i] * 100 for i in top_indices]
    axes[1].barh(bar_labels[::-1], bar_values[::-1], color=["#4CAF50", "#2196F3", "#FF9800"])
    axes[1].set_xlabel("Confidence (%)")
    axes[1].set_title("Top-3 Predictions")
    axes[1].set_xlim(0, 100)
    for i, v in enumerate(bar_values[::-1]):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

    plt.suptitle(f"Predicted: {CLASS_NAMES[top_indices[0]]}", fontsize=13, fontweight="bold", color="green")
    plt.tight_layout()
    plt.show()


import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename(
    title="Select a traffic sign image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.ppm")]
)

if image_path:
    predict_sign(image_path)
else:
    print("No image selected.")
