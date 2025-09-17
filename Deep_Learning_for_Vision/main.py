import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# -----------------------
# PyTorch ResNet18
# -----------------------
resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet_model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Full path to your dog image
dog_img_path = r"C:\Users\micha\OneDrive\Documents\2017codes\CV\Deep_Learning_for_Vision\dog.jpg"
img_pil_dog = Image.open(dog_img_path)
img_tensor = transform(img_pil_dog).unsqueeze(0)

# Predict
with torch.no_grad():
    output = resnet_model(img_tensor)
    _, predicted_idx = output.max(1)

# Get human-readable class name
from torchvision.models import ResNet18_Weights
imagenet_classes = ResNet18_Weights.DEFAULT.meta["categories"]
resnet_class_name = imagenet_classes[predicted_idx.item()]
print(f"PyTorch ResNet18 prediction: {resnet_class_name} (index {predicted_idx.item()})")

# Visualize PyTorch prediction
plt.figure(figsize=(6,6))
plt.imshow(img_pil_dog)
plt.title(f"ResNet18 Prediction: {resnet_class_name}")
plt.axis('off')
plt.show()

# -----------------------
# TensorFlow MobileNetV2
# -----------------------
mobilenet_model = MobileNetV2(weights='imagenet')

# Full path to your cat image
cat_img_path = r"C:\Users\micha\OneDrive\Documents\2017codes\CV\Deep_Learning_for_Vision\cat.png"
img_tf = image.load_img(cat_img_path, target_size=(224, 224))
x = image.img_to_array(img_tf)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = mobilenet_model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

print("TensorFlow MobileNetV2 predictions:")
for i, (imagenet_id, name, prob) in enumerate(decoded_preds, start=1):
    print(f"{i}. {name} ({prob*100:.2f}%)")

# Visualize TensorFlow prediction
plt.figure(figsize=(6,6))
plt.imshow(img_tf)
plt.title(f"MobileNetV2 Top-1 Prediction: {decoded_preds[0][1]}")
plt.axis('off')
plt.show()
