"""
    Author: ARK1375
    Create: 24 Aug 2020
    Mod:    05 Sep 2020 23:11
    Description:

"""

import PIL as pil
from PIL import Image
import numpy as np
from os import listdir
import math as mp
import matplotlib.pyplot as plt
import time

#location of the files
locs = {'loc_train' : r"./data/mnist_png/training/" , 'loc_test' : r"./data/mnist_png/testing/"}

#A function for loading the image as a numpy ndarray
def load(name, address):

    loc = address
    
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
    sections = []
    for sec in secs:
    
        indexs = np.where(sec > tolarance)

        # !lower performance but probably works o.k
        # shape1 = [ ( indexs[0][i], indexs[1][i] ) for i in range( len(indexs[0]) )]

        indexs = np.concatenate(indexs).reshape(2, indexs[0].size).transpose(1,0)
        indexs[:,1] -= mp.ceil(num_pix/2)
        indexs[:,0] *= -1
        indexs[:,0] += mp.ceil(num_pix/2)
        # TODO: We can use the non-zero index here by just adding a 1 to apropriate indexies

        norms = indexs.copy()
        norms **= 2
        norms = np.sum(norms , axis = 1)
        norms = np.concatenate((norms,norms)).reshape(2 , len(norms) ).transpose(1,0)

        normaled = indexs / (np.where(norms == 0 , 1000000 , np.sqrt(norms)))
        fin_vector = np.sum(normaled , axis = 0)

        sections.append(fin_vector)

        #TODO: We can normalize the final vector but because of showing the pixel density propertis, we refused to do that for now
        # fin_vector_norm =  (fin_vector[0]*fin_vector[0] + fin_vector[1]*fin_vector[1])
        # fin_vector /= ( np.where(fin_vector_norm == 0 , 1000000 , fin_vector_norm) )
        
    return np.array(sections)

def get_training_data():

    data , labels = [] , []
    
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)

    for i in range(10):

        address = locs['loc_train'] + f"{i}/"
        ls_dir = listdir(address)

        for name in ls_dir:
            img = load(name, address)
            sections = section(img , factor = 4)
            vecs = calc_vectors(sections, 150)
            out = i
            data.append(vecs.reshape(32))
            labels.append(out)
    
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)

    return np.array(data).astype('float32') , np.array(labels).astype('float32')

def get_testing_data():
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)

    data , labels = [] , []

    for i in range(10):
        
        address = locs['loc_test'] + f"{i}/"
        ls_dir = listdir(address)

        for name in ls_dir:
            img = load(name, address)
            sections = section(img , factor = 4)
            vecs = calc_vectors(sections, 50)
            out = i
            data.append(vecs.reshape(32))
            labels.append(out)
    
    localtime = time.asctime( time.localtime(time.time()) )
    print (localtime)

    return np.array(data).astype('float32') , np.array(labels).astype('float32')

 
# load_ds = get_training_data()
# print(load_ds[0])

# ?Imort one element
# a = load("1.png" , "loc_train")
# sections = section(a , factor = 2)
# se = calc_vectors(sections, 10)
# print(se.shape)

# ?Ploting vectors
# origin = np.zeros(se.shape[0] , dtype = "int16")
# plt.quiver(origin , origin ,se[:,0] , se[:,1] , scale = 1)
# plt.show()

#?Ploting Points
# plt.plot(origin ,se[:,0],se[:,1] ,'ro')
# plt.show()

# ?Imorting
# ls_dir = listdir(locs['loc_train'])
# for i in ls_dir:
#     a = load(i , "loc_train")
#     sections = section(a)
#     se = calc_vectors(sections, 10)

