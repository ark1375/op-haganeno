"""
    Author: ARK1375
    Date:   24 Aug 2020
    Mod:    25 Aug 2020     19:27
    Description:

"""
import PIL as pil
from PIL import Image
import numpy as np
from os import listdir
# from keras.preprocessing.image import load_img



#location of the files

locs = {'loc_train' : r"./data/mnist_png/training/0/" , 'loc_test' : r"./data/mnist_png/testing/0/"}

#A function for loading the image as a numpy ndarray
def load(name, tr_ts):

    loc = locs[tr_ts]
    
    image = Image.open(loc+name)
    data = np.asarray(image)
    #check if the data is valid
    if (data.shape == (28,28)):
        return data

#seting the location of train and test files
def set_locs(trainloc, testloc):
    locs['loc_train'] = trainloc
    locs['loc_test'] = testloc

def section(img , factor = 2):
    secs = []

    if (img.shape[0] % factor != 0):
        return False
    
    pix_size = int(img.shape[0] / factor)

    img_shape = img.shape
    trans_1 = img.reshape(factor, pix_size , factor , pix_size)
    trans_2 = trans_1.transpose(0,2,1,3)
    trans_3 = trans_2.reshape(-1, pix_size, pix_size)

    return trans_3


ls_dir = listdir(locs['loc_train'])
for i in ls_dir:
    a = load(i , "loc_train")
    section(a)

