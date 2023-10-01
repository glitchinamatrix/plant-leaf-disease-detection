#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load the MobileNetV2 pre-trained model (weights pre-trained on ImageNet)
model = MobileNetV2(weights='imagenet')

# Load and preprocess the test image
test_image_path = r"C:\Users\lenovo\OneDrive\Documents\PlantVillage\Pepper__bell___Bacterial_spot\01dfb88b-cd5a-420c-b163-51f5fe07b74d___JR_B.Spot 9091.JPG"
img = cv2.imread(test_image_path)
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MobileNetV2 expects RGB images
img = preprocess_input(img)  # Preprocess the image

# Make predictions on the preprocessed image
predictions = model.predict(np.expand_dims(img, axis=0))
decoded_predictions = decode_predictions(predictions)

# Display the predicted labels and their probabilities
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i + 1}: {label} ({score:.2f})")

# Get the top predicted label and its probability
top_label = decoded_predictions[0][0][1]
top_probability = decoded_predictions[0][0][2]
print(f"Top predicted label: {top_label}")
print(f"Probability: {top_probability:.2f}")


# In[6]:


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt

# Load the MobileNetV2 pre-trained model (weights pre-trained on ImageNet)
model = MobileNetV2(weights='imagenet')

# Load and preprocess the test image
test_image_path = r"C:\Users\lenovo\OneDrive\Documents\PlantVillage\Pepper__bell___Bacterial_spot\01dfb88b-cd5a-420c-b163-51f5fe07b74d___JR_B.Spot 9091.JPG"
img = cv2.imread(test_image_path)
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MobileNetV2 expects RGB images
img = preprocess_input(img)  # Preprocess the image

# Make predictions on the preprocessed image
predictions = model.predict(np.expand_dims(img, axis=0))
decoded_predictions = decode_predictions(predictions)

# Get the top predicted label and its probability
top_label = decoded_predictions[0][0][1]
top_probability = decoded_predictions[0][0][2]

# Display the test image along with the predicted label and probability
plt.imshow(cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB))
plt.title(f"Predicted label: {top_label}\nProbability: {top_probability:.2f}")
plt.axis('off')
plt.show()


# In[ ]:




