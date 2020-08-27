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
import math as mp
import matplotlib.pyplot as plt




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

        normaled = indexs / (np.where(norms == 0 , 1000000 , norms))
        fin_vector = np.sum(normaled , axis = 0)
        sections.append(fin_vector)

        #TODO: We can normalize the final vector but because of showing the pixel density propertis, we refused to do that for now
        # fin_vector_norm =  fin_vector[0]*fin_vector[0] + fin_vector[1]*fin_vector[1]
        # fin_vector /= fin_vector_norm
        
    return np.array(sections)


# ?Imort one element
a = load("1.png" , "loc_train")
sections = section(a , factor = 2)
se = calc_vectors(sections, 10)

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

