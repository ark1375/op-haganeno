"""
    Author: ARK1375
    Create: 27 Aug 2020
    Mod:    05 Sep 2020 23:10
    Description:

"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sectioning
import random
from matplotlib import pyplot as plt
import time

#input shape is simply the image in vector form so for an image sectioned in 16 parts the input would be (16,2) for 16 (x,y) cordinates

def learn_by_sectioning():
    model_a = keras.Sequential()
    model_a.add(keras.Input(shape = (32,)))

    model_a.add(layers.Dense(60, activation='relu', name = "layerA"))
    model_a.add(layers.Dense(60, activation='relu', name = "layerB"))
    model_a.add(layers.Dense(10, activation='softmax', name = "layerC"))

    model_a.summary()

    indx_train = np.arange(0,60000)
    np.random.shuffle(indx_train)

    (x_test , y_test) = sectioning.get_testing_data()
    (x_train , y_train) = sectioning.get_training_data()

    x_val = x_train[indx_train[:10000]]
    y_val = y_train[indx_train[:10000]]
    x_train = x_train[indx_train[10000:]]
    y_train = y_train[indx_train[10000:]]

    model_a.compile(
        optimizer = keras.optimizers.RMSprop(),
        loss = keras.losses.SparseCategoricalCrossentropy(),
        metrics = "acc",
    )

    localtime = time.asctime( time.localtime(time.time()) )
    print (f"\n\nBegin Training:{localtime}")

    history = model_a.fit(
        x_train,
        y_train,
        batch_size = 64,
        epochs = 20,
        validation_data=(x_val , y_val)
    )

    localtime = time.asctime( time.localtime(time.time()) )
    print (f"\n\nEnd Training:{localtime}")

    # print(history.history)

    print("\n\n####\nEvaluate on test data")
    result = model_a.evaluate(x_test , y_test , batch_size = 128)
    print("Test loss , test acc:", result)


    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1 , len(acc) + 1)
    plt.plot(epochs , acc , 'bo' , label = 'training acc')
    plt.plot(epochs , val_acc , 'r--' , label = 'validation acc')
    plt.title("training and validation acc")
    plt.legend()
    plt.figure()
    plt.plot(epochs , loss , 'bo' , label = 'training loss')
    plt.plot(epochs , val_loss , 'r--' , label = 'validation loss')
    plt.title("training and validation loss")
    plt.legend()
    plt.figure()
    plt.show()

    indx_test = np.arange(0,10000)
    np.random.shuffle(indx_test)

    prediction = model_a.predict(x_test[indx_test[:20]])
    print(np.argmax(prediction , axis = 1))
    print(prediction.shape)
    print(f"\n\nlabels = {y_test[indx_test[:20]]}")


def learn_raw():
    # this is simply the default code writen in the keras documentation with minor mods

    model_a = keras.Sequential()
    model_a.add(keras.Input(shape = (784,)))
    model_a.add(layers.Dense(30, activation='relu', name = "layerA"))
    model_a.add(layers.Dense(30, activation='relu', name = "layerB"))
    model_a.add(layers.Dense(10, activation='softmax', name = "layerC"))

    model_a.summary()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255
    
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    model_a.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics='acc',
    )

    localtime = time.asctime( time.localtime(time.time()) )
    print (f"\n\nBegin Training:{localtime}")

    print("Fit model on training data")
    history = model_a.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=20,
        validation_data=(x_val, y_val),
    )

    localtime = time.asctime( time.localtime(time.time()) )
    print (f"\n\nEnd Training:{localtime}")
    
    print("Evaluate on test data")
    results = model_a.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1 , len(acc) + 1)
    plt.plot(epochs , acc , 'bo' , label = 'training acc')
    plt.plot(epochs , val_acc , 'r--' , label = 'validation acc')
    plt.title("training and validation acc")
    plt.legend()
    plt.figure()
    plt.plot(epochs , loss , 'bo' , label = 'training loss')
    plt.plot(epochs , val_loss , 'r--' , label = 'validation loss')
    plt.title("training and validation loss")
    plt.legend()
    plt.figure()
    plt.show()

    indx_test = np.arange(0,10000)
    np.random.shuffle(indx_test)

    prediction = model_a.predict(x_test[indx_test[:10]])
    print(np.argmax(prediction , axis = 1))
    print(prediction.shape)
    print(f"\n\nlabels = {y_test[indx_test[:10]]}")









