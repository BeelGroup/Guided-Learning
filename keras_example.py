from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard

epochs = 5000

def train(inputs, expected_outputs):

    model = Sequential()
    model.add(Dense(inputs.shape[-1], activation='sigmoid', input_shape=(inputs.shape[-1],))) # input layer
    model.add(Dense(25, activation='sigmoid'))
    model.add(Dense(expected_outputs.shape[-1], activation='sigmoid')) # output layer

    # 25 is best so far @ 5000 epochs (2.1854e-04 loss)

    model.summary()

    model.compile(loss='mean_squared_error',
                  optimizer=SGD(lr=0.1),
                  metrics=['accuracy'])

    tbCallBack = TensorBoard(log_dir="./logs", histogram_freq=0, write_graph=True, write_images=True)
    model.fit(inputs, expected_outputs,
              epochs=epochs,
              batch_size=4,
              verbose=1,
              callbacks=[tbCallBack])

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

    return model
