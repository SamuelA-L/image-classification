import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


def picture_examples():
    roses = list(data_dir.glob('roses/*'))
    img1 = PIL.Image.open(str(roses[0]))

    plt.imshow(img1)
    plt.show()

    tulips = list(data_dir.glob('tulips/*'))
    img2 = PIL.Image.open(str(tulips[0]))
    plt.imshow(img2)

    plt.show()


batch_size = 32
img_height = 180
img_width = 180

training_set = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=8,
  image_size=(img_height, img_width),
  batch_size=batch_size)


validation_set = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=8,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = training_set.class_names
print(class_names)


def data_vis():
    plt.figure(figsize=(10, 10))
    for images, labels in training_set.take(1):
      for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()


for image_batch, labels_batch in training_set:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


AUTOTUNE = tf.data.AUTOTUNE

train_ds = training_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = validation_set.cache().prefetch(buffer_size=AUTOTUNE)


normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)


num_classes = len(class_names)

model = Sequential([
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

def train():

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()


    epochs=15
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
    )
    model.save('models/model1')


    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    return model


def predict(path, model):

    img = keras.preprocessing.image.load_img(
        path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

model = train()
# model = keras.models.load_model('models/model1')

predict('/home/samuel/Desktop/daisies.jpg', model)