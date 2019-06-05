import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras import models, optimizers
import time
from keras.models import model_from_json
from os import listdir
from os.path import isfile, join

model = models.load_model("cars_cnn_b.h5")
model.compile(loss='categorical_crossentropy',
 optimizer=optimizers.RMSprop(lr=1e-4),
 metrics=['categorical_accuracy'])

brands = ["Audi A4", "BMW 3", "VW Golf", "Mercedes E", "Opel Astra"]

mypath = "/cars_dataset/test/audi"
imgs = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(imgs)
golf = 0
bmw = 0
merc = 0
audi = 0
opel = 0
for i in imgs:
    fname = i

    test_img = load_img("/cars_dataset/test/audi/"+fname, target_size=(150, 150))
    test_img = img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis=0)
    result = model.predict(test_img)

    try:
        id = list(result[0]).index(1.)
    except:
        print(result)
    if id == 0:
        audi += 1
    if id == 1:
        bmw += 1
    if id == 2:
        golf += 1
    if id == 3:
        merc += 1
    if id == 4:
        opel += 1
print(audi)
print(bmw)
print(golf)
print(merc)
print(opel)