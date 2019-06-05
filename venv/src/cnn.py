import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import load_img, img_to_array


#Start
train_data_path = '/cars_dataset/train'
test_data_path = '/cars_dataset/validation'
img_rows = 150
img_cols = 150
epochs = 30
batch_size = 32
num_of_train_samples = 1000
num_of_test_samples = 500

#Image Generator
# train_datagen = ImageDataGenerator(rescale=1. / 255,
#                                    rotation_range=40,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True,
#                                    fill_mode='nearest')

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(test_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')

# Build model
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_rows, img_cols, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#Train
history = model.fit_generator(train_generator,
                    steps_per_epoch=num_of_train_samples // batch_size,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_of_test_samples // batch_size)

model_json = model.to_json()
with open("cars_cnn_b.json", "w") as file:
    file.write(model_json)
model.save("cars_cnn_b.h5")
model.save_weights("cars_weights_b.h5")

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

# #Confution Matrix and Classification Report
# Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size+1)
# y_pred = np.argmax(Y_pred, axis=1)
# print('Confusion Matrix')
# print(len(validation_generator.classes))
# # print(confusion_matrix(validation_generator.classes, y_pred))
# # print('Classification Report')
# # target_names = ['audi', 'bmw', 'golf']
# # print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
#
# brands = ["Audi A4", "BMW 3", "VW Golf"]

# def predd(fname):
#     test_img = load_img("/cars_dataset/test/"+fname, target_size=(150, 150))
#     test_img = img_to_array(test_img)
#     test_img = np.expand_dims(test_img, axis=0)
#     result = model.predict(test_img)
#     id = list(result[0]).index(1.)
#
#     img = mpimg.imread("/cars_dataset/test/"+fname)
#     imgplot = plt.imshow(img)
#     plt.figtext(.5, .95, brands[id] , fontsize=20, ha='center')
#
#     plt.show()
#     print(brands[id])
#
# predd("golf/00039.jpg")
# predd("golf/00046.jpg")
# predd("golf/00050.jpg")
# predd("audi_a4/00011.jpg")
# predd("audi_a4/00017.jpg")
# predd("audi_a4/00036.jpg")
# predd("bmw_seria3/00025.jpg")
# predd("bmw_seria3/00059.jpg")
# predd("bmw_seria3/00067.jpg")