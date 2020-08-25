#   Author ARK1375
#   Date: Aug 24 2020
#
#   Summary:
#

import PIL as pil
from PIL import Image

loc = r"./data/single_data/"

def load(name):
    image = Image.open(loc+name)
    return image.getdata()

print(  len(    list(load("id_5_label_4.png")   )  ) )

