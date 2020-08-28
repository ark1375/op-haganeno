"""
    Author: ARK1375
    Create: 27 Aug 2020
    Mod:    27 Aug 2020 23:09
    Description:

"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model(input_shape):
    #input shape is simply the image in vector form so for an image sectioned in 16 parts the input would be (16,2) for 16 (x,y) cordinates
    model = keras.Sequential()
    model.add(keras.Input(shape = input_shape))
    model.add(layers.Dense(20, activation='sigmoid'))
    model.add(layers.Dense(20, activation='sigmoid'))
    model.add(layers.Dense(20, activation='sigmoid'))
    model.add(layers.Dense(10, activation='sigmoid'))
    model.summary()

    return model

create_model((16,))