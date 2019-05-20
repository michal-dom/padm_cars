from urllib.request import Request, urlopen, urlretrieve
from PIL import Image
import re
import time
import os

links = []

file_links = "audi_links.txt"
img_folder = "audi_img/"

os.mkdir(img_folder)

with open(file_links, "r") as myfile:
    links = myfile.readlines()

del links[::2]

i = 0
for link in links:
    req = Request(
        link,
        headers={'User-Agent': 'Mozilla/5.0'})

    mybytes = urlopen(req).read()
    mystr = mybytes.decode("utf8")
    reg = re.findall(r"<img[^>]* data-lazy=\"([^\"]*)\"",
        mystr,
        re.DOTALL)


    for img in reg[:4]:
        try:
            i += 1
            # time.sleep(1)
            requ = Request(
                img,
                headers={'User-Agent': 'Mozilla/5.0'})
            img_from_web = Image.open(urlopen(requ))
            w, h = img_from_web.size
            new_width = int(w / 2)
            new_height = int((h - 40) / 2)

            img_from_web.crop((0, 0, w, h - 40)).resize((new_width, new_height), resample=Image.BICUBIC).save(
                img_folder+"000" + str(i) + ".jpg")

            print(img)
        except Exception as e:
            print(e)



