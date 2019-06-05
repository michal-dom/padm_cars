from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras import models
from keras import layers
from keras import optimizers

import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


conv_base = VGG16(weights='imagenet',
 include_top=False,
 input_shape=(150, 150, 3))

# conv_base.summary()


base_dir = '/cars_dataset'

train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=32, class_mode='categorical')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 40

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count, 5))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')
    i = 0

    # print(generator.)

    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        print(features_batch.shape)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        print(labels_batch.shape)
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch

        i += 1
        if i * batch_size >= sample_count:
            break
    return  features, labels

train_features, train_labels = extract_features(train_dir, 5000)
validation_features, validation_labels = extract_features(valid_dir, 2500)
test_features, test_labels = extract_features(test_dir, 2500)

train_features = np.reshape(train_features, (5000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (2500, 4 * 4 * 512))
test_features = np.reshape(test_features, (2500, 4 * 4 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
 loss='categorical_crossentropy',
 metrics=['acc'])

history = model.fit(train_features, train_labels,
 epochs=30,
 batch_size=20,
 validation_data=(validation_features, validation_labels))

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('dokładność podczas testowania:', test_acc)
print('strata podczas testowania:', test_loss)



try:
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
    plt.plot(epochs, val_acc, 'b', label='Dokladnosc walidacji')
    plt.title('Dokladnosc trenowania i walidacji')

    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Strata trenowania')
    plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
    plt.title('Strata trenowania i walidacji')
    plt.legend()
    plt.show()

except:
    print('oo')