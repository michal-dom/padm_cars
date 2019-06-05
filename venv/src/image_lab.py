import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras import models, optimizers
import time
from keras.models import model_from_json

json_file = open('cars_cnn.json', 'r')
loaded_model_json = json_file.read()

json_file.close()

# model = model_from_json(loaded_model_json)
# model.load_weights('cars_weights.h5')
model = models.load_model("cats_and_dogs_small_1.h5")
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
 optimizer=optimizers.RMSprop(lr=1e-4),
 metrics=['categorical_accuracy'])

brands = ["Audi A4", "BMW 3", "VW Golf"]

def predd(fname):
    test_img = load_img("/cars_dataset/test/"+fname, target_size=(150, 150))
    test_img = img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    start = int(round(time.time() * 1000))

    result = model.predict(test_img)
    end = int(round(time.time() * 1000))
    print(end-start)
    id = list(result[0]).index(1.)

    img = mpimg.imread("/cars_dataset/test/"+fname)
    imgplot = plt.imshow(img)
    plt.figtext(.5, .95, brands[id] , fontsize=20, ha='center')

    plt.show()
    print(brands[id])

predd("golf/00039.jpg")
predd("golf/00046.jpg")
predd("golf/00050.jpg")
predd("audi_a4/00011.jpg")
predd("audi_a4/00017.jpg")
predd("audi_a4/00036.jpg")
predd("bmw_seria3/00025.jpg")
predd("bmw_seria3/00059.jpg")
predd("bmw_seria3/00067.jpg")


