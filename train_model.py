import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import cv2
import numpy as np

# =========================================
# DATASET PATHS
# =========================================

train_dir = "dataset/train"
test_dir = "dataset/test"

# =========================================
# DATA AUGMENTATION
# =========================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    width_shift_range=0.2,
    height_shift_range=0.2
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# =========================================
# TRAIN GENERATOR
# =========================================

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

# =========================================
# TEST GENERATOR
# =========================================

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode="categorical"
)

# =========================================
# BUILD CNN MODEL
# =========================================

model = Sequential()

# -----------------------------------------
# FIRST CNN BLOCK
# -----------------------------------------

model.add(Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(48,48,1)
))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

# -----------------------------------------
# SECOND CNN BLOCK
# -----------------------------------------

model.add(Conv2D(
    64,
    (3,3),
    activation='relu'
))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

# -----------------------------------------
# THIRD CNN BLOCK
# -----------------------------------------

model.add(Conv2D(
    128,
    (3,3),
    activation='relu'
))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2)))

# =========================================
# FLATTEN
# =========================================

model.add(Flatten())

# =========================================
# DENSE LAYER
# =========================================

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

# =========================================
# OUTPUT LAYER
# =========================================

model.add(Dense(7, activation='softmax'))

# =========================================
# COMPILE MODEL
# =========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================
# EARLY STOPPING
# =========================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# =========================================
# TRAIN MODEL
# =========================================

history = model.fit(
    train_generator,
    epochs=25,
    validation_data=test_generator,
    callbacks=[early_stop]
)

# =========================================
# SAVE MODEL
# =========================================

model.save("model.h5")

print("\nModel Saved Successfully")

# =========================================
# ACCURACY GRAPH
# =========================================

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

plt.show()

# =========================================
# LOSS GRAPH
# =========================================

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend(['Train', 'Validation'])

plt.show()