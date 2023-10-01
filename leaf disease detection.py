#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install opencv-python')
import numpy as np
import pickle
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# In[27]:


import zipfile as zf
files = zf.ZipFile("archive.zip", 'r')
files.extractall('directory to extract')
files.close()


# In[45]:


IMAGE_SIZE=256
BATCH_SIZE=32
import tensorflow as tf
images_dataset= tf.keras.utils.image_dataset_from_directory(
 'dataset',
 shuffle=True,
 image_size=(IMAGE_SIZE,IMAGE_SIZE),
 batch_size=BATCH_SIZE,
 
)


# In[43]:


class_names=images_dataset.class_names
class_names


# In[46]:


len(images_dataset)


# In[4]:


get_ipython().system('pip install keras')
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
IMAGE_SIZE=256
BATCH_SIZE=32
images_dataset= tf.keras.utils.image_dataset_from_directory(
 'dataset',
 shuffle=True,
 image_size=(IMAGE_SIZE,IMAGE_SIZE),
 batch_size=BATCH_SIZE,
 
)
for image_batch, label_batch in images_dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())


# In[5]:


#printing first image of data set
for image_batch, label_batch in images_dataset.take(1):
    print(image_batch[0].numpy())


# In[28]:


#visualize the first image in that batch
get_ipython().system('pip install keras')
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
IMAGE_SIZE=256
BATCH_SIZE=32
images_dataset= tf.keras.utils.image_dataset_from_directory(
 'dataset',
 shuffle=True,
 image_size=(IMAGE_SIZE,IMAGE_SIZE),
 batch_size=BATCH_SIZE,
 
)
class_names=images_dataset.class_names
class_names
plt.figure(figsize=(18,18))
for image_batch, label_batch in images_dataset.take(1):
    for i in range (12):
        ax=plt.subplot(3,4, i+1)
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        plt.title(class_names[label_batch[i]])
        plt.axis('off')
def get_dataset_partitions_tf(ds,train_split=0.8, val_split=0.1, test_split=0.1,shuffle=True, shuffle_size=10000):
    ds_size=len(ds)
    if shuffle:
        ds=ds.shuffle(shuffle_size,seed=12)
        
    train_size= int(train_split* ds_size)
    val_size=int(val_split* ds_size)
    
    train_ds=ds.take(train_size)
    
    val_ds=ds.skip(train_size).take(val_size)
    
    test_ds=ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds
train_ds, val_ds, test_ds=get_dataset_partitions_tf(images_dataset)
print(len(train_ds),len(val_ds),len(test_ds))
train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds=val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds=test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# In[1]:


EPOCHS = 25
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root = 'dataset'
width=256
height=256
depth=3
from tensorflow.keras.utils import img_to_array
from keras.optimizers import Adam
from keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
image_list, label_list = [], []
create_labels('dataset')
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)
print(label_binarizer.classes_)
np_image_list = np.array(image_list, dtype=np.float16) / 225.0
print("[INFO] Spliting data to train, test")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42) 
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")
model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))
model.summary()
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print("[INFO] training network...")
history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1
    )
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")
print("[INFO] Saving model...")
pickle.dump(model,open('cnn_model.pkl', 'wb'))
loaded_model = pickle.load(open('cnn_model.pkl', 'rb'))
loaded_model = pickle.load(open('path\\cnn_model.pkl', 'rb'))
image_dir="path\\plantdisease_dataset\\PlantVillage\\Potato___Early_blight"

im=convert_image_to_array(image_dir)
np_image_li = np.array(im, dtype=np.float16) / 225.0
npp_image = np.expand_dims(np_image_li, axis=0)
result=model.predict(npp_image)

print(result)
itemindex = np.where(result==np.max(result))
print("probability:"+str(np.max(result))+"\n"+label_binarizer.classes_[itemindex[1][0]])


# In[ ]:


get_ipython().system('pip install tensorflow-hub')
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define the path to the pre-trained model
model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

# Load the pre-trained model from TensorFlow Hub
model = tf.keras.Sequential([hub.KerasLayer(model_url)])

# Define the input image size expected by the model
input_image_size = (224, 224)

# Define a function to preprocess an image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize(input_image_size)
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = image[np.newaxis, ...]  # Add batch dimension
    return image

# Define the path to your test image
image_path = r"C:\Users\lenovo\OneDrive\Desktop\PlantVillage\Pepper__bell___Bacterial_spot\00f2e69a-1e56-412d-8a79-fdce794a17e4___JR_B.Spot 3132.JPG"
# Preprocess the test image
test_image = preprocess_image(image_path)

# Make predictions
predictions = model.predict(test_image)

# Apply the softmax function to normalize the predictions
normalized_predictions = tf.nn.softmax(predictions, axis=-1)

top_predictions = predictions[:, 1:1001]
# Decode the predictions to get class labels and probabilities
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(top_predictions, top=5)[0]

# Display the test image and top predicted classes with probabilities
plt.figure(figsize=(8, 8))
plt.imshow(test_image[0])
plt.axis("off")
plt.show()

# Print the top predicted classes and their probabilities
for i, (class_id, class_name, probability) in enumerate(decoded_predictions):
    print(f"Top {i + 1}: Class={class_name}, Probability={probability:.4f}")


# In[ ]:


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





# In[ ]:




