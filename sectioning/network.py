"""
    Author: ARK1375
    Create: 27 Aug 2020
    Mod:    04 Sep 2020 19:49
    Description:

"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sectioning
import random

#input shape is simply the image in vector form so for an image sectioned in 16 parts the input would be (16,2) for 16 (x,y) cordinates
model_a = keras.Sequential()
model_a.add(keras.Input(shape = (16,)))

model_a.add(layers.Dense(30, activation='sigmoid', name = "layerA"))
model_a.add(layers.Dense(30, activation='sigmoid', name = "layerB"))
model_a.add(layers.Dense(10, activation='sigmoid', name = "layerC"))

model_a.summary()

indx_test = np.arange(0,60000)
np.random.shuffle(indx_test)

(x_test , y_test) = sectioning.get_testing_data()
print(indx_test[0:100])
