from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imutils import paths
import numpy as np

import argparse
import imutils
import cv2
import os
import re

base_dir = '/cars_dataset'

train_dir = os.path.join(base_dir, 'train')
train_audi_dir = os.path.join(train_dir, 'audi_a4')
train_bmw_dir = os.path.join(train_dir, 'bmw_seria3')
train_golf_dir = os.path.join(train_dir, 'golf')
train_merc_dir = os.path.join(train_dir, 'mercedes_e')
train_opel_dir = os.path.join(train_dir, 'opel_astra')

valid_dir = os.path.join(base_dir, 'validation')
valid_audi_dir = os.path.join(valid_dir, 'audi_a4')
valid_bmw_dir = os.path.join(valid_dir, 'bmw_seria3')
valid_golf_dir = os.path.join(valid_dir, 'golf')
valid_merc_dir = os.path.join(valid_dir, 'mercedes_e')
valid_opel_dir = os.path.join(valid_dir, 'opel_astra')

test_dir = os.path.join(base_dir, 'test')
test_audi_dir = os.path.join(test_dir, 'audi_a4')
test_bmw_dir = os.path.join(test_dir, 'bmw_seria3')
test_golf_dir = os.path.join(test_dir, 'golf')
test_merc_dir = os.path.join(test_dir, 'mercedes_e')
test_opel_dir = os.path.join(test_dir, 'opel_astra')

def image_to_vector(image, size=(32,32)):
    return cv2.resize(image, size).flatten()

def extract_color(image, bins = (8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)


    return hist.flatten()



imagePaths = list(paths.list_images(base_dir))

data = []
labels = []
# lab = "/cats_and_dogs_small\\test\cats\cat.1500.jpg"
# print(lab.split("\\")[3])
for (i, imagePath) in enumerate(imagePaths):
    if "test_train" in imagePath:
        continue
    animal = imagePath.split("\\")[2]
    # print(animal)
    image = cv2.imread(imagePath)
    hist = extract_color(image)

    data.append(hist)
    labels.append(animal)


(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, labels, test_size=0.25, random_state=42)

params = {"n_neighbors": np.arange(1, 31, 2),
	"metric": ["euclidean", "cityblock"]}

# print("[INFO] tuning hyperparameters via grid search")
model = KNeighborsClassifier(n_jobs=-1)
# grid = GridSearchCV(model, params)
# grid.fit(trainData, trainLabels)
#
# # evaluate the best grid searched model on the testing data
# acc = grid.score(testData, testLabels)
# print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
# print("[INFO] grid search best parameters: {}".format(
#     grid.best_params_))
# # acc = model.score(testFeat, testLabels)

grid = RandomizedSearchCV(model, params)
grid.fit(trainData, trainLabels)
predLabels = grid.predict(testData)

# conf = confusion_matrix(testLabels, predLabels, [0, 1])
# print(conf)

golfA = 0
bmwA = 0
mercA = 0
audiA = 0
opelA = 0

golfB = 0
bmwB = 0
mercB = 0
audiB = 0
opelB = 0

golfC = 0
bmwC = 0
mercC = 0
audiC = 0
opelC = 0

golfD = 0
bmwD = 0
mercD = 0
audiD = 0
opelD = 0

golfE = 0
bmwE = 0
mercE = 0
audiE = 0
opelE = 0

print(testLabels)
print(len(testLabels))
print(predLabels)
print(len(predLabels))

for i in range(2500):
    if testLabels[i] == "audi_a4":
        if predLabels[i] == "audi_a4":
            audiA += 1
        if predLabels[i] == "bmw_seria3":
            bmwA += 1
        if predLabels[i] == "golf":
            golfA += 1
        if predLabels[i] == "mercedes_e":
            mercA += 1
        if predLabels[i] == "opel_astra":
            opelA += 1
    if testLabels[i] == "bmw_seria3":
        if predLabels[i] == "audi_a4":
            audiB += 1
        if predLabels[i] == "bmw_seria3":
            bmwB += 1
        if predLabels[i] == "golf":
            golfB += 1
        if predLabels[i] == "mercedes_e":
            mercB += 1
        if predLabels[i] == "opel_astra":
            opelB += 1
    if testLabels[i] == "golf":
        if predLabels[i] == "audi_a4":
            audiC += 1
        if predLabels[i] == "bmw_seria3":
            bmwC += 1
        if predLabels[i] == "golf":
            golfC += 1
        if predLabels[i] == "mercedes_e":
            mercC += 1
        if predLabels[i] == "opel_astra":
            opelC += 1
    if testLabels[i] == "mercedes_e":
        if predLabels[i] == "audi_a4":
            audiD += 1
        if predLabels[i] == "bmw_seria3":
            bmwD += 1
        if predLabels[i] == "golf":
            golfD += 1
        if predLabels[i] == "mercedes_e":
            mercD += 1
        if predLabels[i] == "opel_astra":
            opelD += 1
    if testLabels[i] == "opel_astra":
        if predLabels[i] == "audi_a4":
            audiE += 1
        if predLabels[i] == "bmw_seria3":
            bmwE += 1
        if predLabels[i] == "golf":
            golfE += 1
        if predLabels[i] == "mercedes_e":
            mercE += 1
        if predLabels[i] == "opel_astra":
            opelE += 1

print(audiA)
print(bmwA)
print(golfA)
print(mercA)
print(opelA)
print()
print(audiB)
print(bmwB)
print(golfB)
print(mercB)
print(opelB)
print()
print(audiC)
print(bmwC)
print(golfC)
print(mercC)
print(opelC)
print()
print(audiD)
print(bmwD)
print(golfD)
print(mercD)
print(opelD)
print()
print(audiE)
print(bmwE)
print(golfE)
print(mercE)
print(opelE)

# evaluate the best randomized searched model on the testing
# data
acc = grid.score(testData, testLabels)
print("[INFO] grid search accuracy: {:.2f}%".format(acc * 100))
print("[INFO] randomized search best parameters: {}".format(
    grid.best_params_))

