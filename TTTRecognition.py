"""
Very basic image recognition to distinguish between O's and X's
"""
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
from PIL import Image
import cv2
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt

#TODO: add empty squares

total_images = []
total_labels = np.array([0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1,
                          1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1,
                          1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
                          0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,])

total_count = len(total_labels)
train_count = int(total_count * 0.8)

# TODO: combine this into a single loop

# load all the images
for i in range(len(total_labels)):
    img = cv2.imread("./Images/Output"+ str(i) + ".jpg", 0)
    # TODO: crop edges
    numpydata = asarray(img)
    total_images.append(numpydata)

total_images = np.array(total_images)

# Shuffle images
p = np.random.permutation(total_count)

total_images = total_images[p]
total_labels = total_labels[p]

# Diving into training and testing
print(total_images.shape)
train_images = total_images[0:train_count]
train_labels = total_labels[0:train_count]
test_images = total_images[train_count:]
test_labels = total_labels[train_count:]


#normalize
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Create model to test with this
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(20, 30)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20)

# Save model
model.save_weights('./Checkpoints/TTT_Checkpoint')


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
