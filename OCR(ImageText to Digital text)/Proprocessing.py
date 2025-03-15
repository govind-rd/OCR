import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define preprocessing function
def preprocess_image(img):
    # Convert to grayscale if the image is colored
    if len(img.shape) == 3:  # If the image has more than 2 dimensions (RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image to target size (128x32)
    img = cv2.resize(img, (32, 128))

    # Normalize pixel values to the range [0, 1]
    img = img / 255.0

    # Optionally apply thresholding (useful for improving contrast)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Expand dimensions to add channel dimension (for grayscale images)
    img = np.expand_dims(img, axis=-1)

    return img

# Augmentation configuration for data augmentation
def get_data_augmentation():
    datagen = ImageDataGenerator(
        rotation_range=10,       # Random rotation up to 10 degrees
        width_shift_range=0.1,   # Random horizontal shift
        height_shift_range=0.1,  # Random vertical shift
        zoom_range=0.1,          # Random zoom
        shear_range=0.1,         # Random shear
        horizontal_flip=True,    # Random horizontal flip
        fill_mode='nearest'      # Fill mode for pixels that get cropped or transformed
    )
    return datagen

# Function to apply augmentation and preprocessing to data
def preprocess_and_augment(images, labels, batch_size=32):
    images = np.array([preprocess_image(img) for img in images])

    # Apply augmentation
    datagen = get_data_augmentation()
    datagen.fit(images)

    # Convert labels to integer sequences based on the character set
    label_sequences = [[characters.index(c) for c in label] for label in labels]
    return images, label_sequences, datagen

# Example with dummy data
train_images = np.random.rand(100, 128, 32)  # Dummy images
train_labels = ["hello" for _ in range(100)]  # Dummy labels

train_images, train_label_sequences, datagen = preprocess_and_augment(train_images, train_labels)

# Data generator for training
def data_generator(images, label_sequences, batch_size=32):
    for i in range(0, len(images), batch_size):
        x_batch = images[i:i+batch_size]
        y_batch = tf.keras.preprocessing.sequence.pad_sequences(
            label_sequences[i:i+batch_size], padding="post"
        )
        yield x_batch, y_batch

# Training the model
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_images, train_label_sequences),
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, 128, 32, 1), (None, None))
)

model.fit(train_dataset, epochs=50, steps_per_epoch=len(train_images)//32)
