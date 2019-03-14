from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.models import load_model

import numpy as np

import neat, visualize

import h5py, json

import sys, random

epochs = 5000

def train_single_shot(inputs, expected_outputs):
    # Train on the first sample given
    assert(inputs.shape[0]==1 and expected_outputs.shape[0]==1, "")

    model = Sequential()
    model.add(Dense(15, activation='sigmoid')) # single hidden layer
    #model.add(Dense(3, activation='sigmoid'))
    #model.add(Dense(3, activation='sigmoid'))
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

    print("[train_single_shot] inputs: {}".format(inputs))
    print("[train_single_shot] inputs.shape: {}".format(inputs.shape))
    print("[train_single_shot] expected_outputs: {}".format(expected_outputs))
    predictions = model.predict(inputs, verbose=1)
    for i, p in enumerate(predictions):
        print(p)
        print(expected_outputs[i])
    score = model.evaluate(inputs, expected_outputs, verbose=0)
    print('[train_single_shot] Test loss:', score[0])
    print('[train_single_shot] Test accuracy:', score[1])

    print("[train_single_shot] Saving model..")
    save_model(model)
    print("[train_single_shot] Saved model to disk")

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


def keras2neat(neat_config, keras_h5_path, new_genome_key, run_name='1'):
    new_genome_key = str(new_genome_key)

    print("[keras2neat] Loading model data..")
    model_data = load_keras2neat_model_data(keras_h5_path)
    print("[keras2neat] Loaded.")

    # Use a random key
    new_genome = neat.DefaultGenome(new_genome_key)

    num_inputs = model_data[0]['kernel'].shape[0]
    num_outputs = model_data[-1]['kernel'].shape[1]

    new_node_id = num_outputs
    prev_layer_size = None
    total_layer_size = 0
    prev_layer_first_id = 0
    prev_layer_last_id = 0
    layer_last_id = 0
    complete = False

    # create the output nodes
    for o in range(num_outputs):
        new_node = new_genome.create_node(neat_config.genome_config, o)
        # set the bias of the node
        setattr(new_node, 'bias', model_data[-1]["bias"][o])
        new_genome.nodes[o] = new_node

    for layer_id, layer in enumerate(model_data):
        if complete:
            break
        layer_size = layer["bias"].shape[0]
        layer_first_node_id = new_node_id
        total_layer_size += layer_size
        if layer_id != 0:
            prev_layer_first_id = new_node_id - prev_layer_size
            prev_layer_last_id = prev_layer_first_id + prev_layer_size
            layer_last_id = layer_first_node_id + layer_size
        for j in range(layer_size):
            # each entry in ['bias'] represents a node in this layer
            # create a new node
            new_node = new_genome.create_node(neat_config.genome_config, new_node_id)
            # set the bias of the node
            setattr(new_node, 'bias', layer["bias"][j])
            # add the node to the genome
            new_genome.nodes[new_node_id] = new_node

            # add the connections to the previous layer. NOTE: Assumes a fully connected network
            if layer_id == 0:
                # connect to the inputs (ID is negative)
                for k in range(-1, -num_inputs - 1, -1):
                    #print("connecting {} to {}".format(k, new_node_id))
                    new_genome.add_connection(neat_config.genome_config, k, new_node_id, layer["kernel"][k][j],
                                              True)
            elif layer_id == len(model_data) - 1:
                # connect the outputs
                for p in range(prev_layer_size):
                    for o in range(num_outputs):
                        # connect the prev_layer to the output
                        new_genome.add_connection(neat_config.genome_config, p + prev_layer_first_id, o,
                                                  layer['kernel'][p][o], True)
                complete = True
                break
            else:
                #print("Connecting hidden layer {}..".format(layer_id))
                # connect to the previous layer
                for k_i, k in enumerate(layer['kernel']):
                    new_genome.add_connection(neat_config.genome_config, k_i + prev_layer_first_id, new_node_id,
                                              k[j], True)
            new_node_id += 1
            prev_layer_size = layer_size

    visualize.draw_net(neat_config, new_genome, view=False, filename="eval/run_{}/keras2neat/{}".format(run_name ,new_genome_key))

    return new_genome