"""
    Author: ARK1375
    Date:   24 Aug 2020
    Mod:    26 Aug 2020     23:27
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

    if (img.shape[0] % factor != 0):
        return False
    
    pix_size = int(img.shape[0] / factor)

    trans_1 = img.reshape(factor, pix_size , factor , pix_size)
    trans_2 = trans_1.transpose(0,2,1,3)
    trans_3 = trans_2.reshape(-1, pix_size, pix_size)

    return trans_3

def calc_vectors(secs , tolarance):

    num_pix = secs[0].shape[0]

    for sec in secs:
    
        indexs = np.where(sec > tolarance)
        
        # lower performance but probably works o.k
        # shape = [ ( indexs[0][i], indexs[1][i] ) for i in range( len(indexs[0]) )]
        shape = np.concatenate(indexs).reshape(2, indexs[0].size).transpose(1,0)


        print(shape)

    return shape


a = load("1.png" , "loc_train")
se = calc_vectors(section(a), 10)

# ls_dir = listdir(locs['loc_train'])
# for i in ls_dir:
#     a = load(i , "loc_train")
#     section(a)

