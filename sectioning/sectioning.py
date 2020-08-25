"""
    Author: ARK1375
    Date:   24 Aug 2020
    Mod:    25 Aug 2020     19:27
    Description:

"""
import PIL as pil
from PIL import Image
import numpy as np



#location of the files

locs = {'loc_train' : r"./data/single_data/train/" , 'loc_test' : r"./data/single_data/test/"}

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

print(load("id_5_label_4.png" , "loc_train"))


