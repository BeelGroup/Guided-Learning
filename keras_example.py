from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.models import load_model

import numpy as np

import h5py, json

import sys

epochs = 5000

def train(inputs, expected_outputs):
    print("[train] shape of inputs: {}".format(inputs.shape))
    print("[train] shape of outputs: {}".format(expected_outputs.shape))

    model = Sequential()
    #model.add(Dense(25, activation='sigmoid')) # single hidden layer
    model.add(Dense(3, activation='sigmoid'))
    model.add(Dense(3, activation='sigmoid'))
    model.add(Dense(expected_outputs.shape[-1], activation='sigmoid')) # output layer

    # 25 is best so far @ 5000 epochs (2.1854e-04 loss)

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    tbCallBack = TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)
    model.fit(inputs, expected_outputs,
              epochs=epochs,
              batch_size=4,
              verbose=1,
              callbacks=[tbCallBack])

    model.summary()

    print("[keras_example] inputs: {}".format(inputs))
    print("[keras_example] inputs.shape: {}".format(inputs.shape))
    print("[keras_example] expected_outputs: {}".format(expected_outputs))
    predictions = model.predict(inputs, verbose=1)
    for i, p in enumerate(predictions):
        print(p)
        print(expected_outputs[i])
    score = model.evaluate(inputs, expected_outputs, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print("Saving model..")
    save_model(model)
    print("Saved model to disk")

    return model


def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")

def load_keras2neat_model_data(hdf5_weight_path):
    '''
    :param hdf5_weight_path: The path to the Keras HDF5 weight file for a fully dense network
    :return: An associative array of Numpy arrays of weights
    '''
    ret = [] # The last entry will always be the output layer

    f = h5py.File(hdf5_weight_path, 'r')
    for i, key in enumerate(f):
        print("[load_keras2neat_model_data] {}".format(f[key]))
        for k in f[key]:
            print("[load_keras2neat_model_data] {}".format(f[key][k]))
            bias = np.array(f[key][k]['bias:0'])
            kernel = np.array(f[key][k]['kernel:0'])

            temp = {}
            temp["bias"] = bias
            temp["kernel"] = kernel

            ret.append(temp)

    return ret