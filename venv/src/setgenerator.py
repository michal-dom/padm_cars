import random
from os import listdir, mkdir
from shutil import copyfile

train_folder = "/cars_dataset/train"
test_folder = "/cars_dataset/test"
valid_folder = "/cars_dataset/validation"

car = "/opel_astra/"

train_folder = train_folder + car
test_folder = test_folder + car
valid_folder = valid_folder + car

mkdir(train_folder)
mkdir(test_folder)
mkdir(valid_folder)

folder = "astra_data/"

img_list = listdir(folder)

train_set = random.sample(img_list, 1000)

rest = [item for item in img_list if item not in set(train_set)]

valid_set = random.sample(rest, 500)

test_set = [item for item in rest if item not in set(valid_set)]

print(len(set(train_set)))
print(len(set(valid_set)))
print(len(set(test_set)))

for img in train_set:
    copyfile(folder + img, train_folder+img)

for img in test_set:
    copyfile(folder + img, test_folder+img)

for img in valid_set:
    copyfile(folder + img, valid_folder+img)