#import necessary libraries but also useful previously defined functions 
from SHodge_learn import data_wrangle_S
from SHodge_learn import daattavya_accuracy

from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import numpy as np
import requests
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request

################################################################################
#define new architecture for the NN, same as in prep_work.py but with an additional, "summed" layer
def equivariant_layer(inp, number_of_channels_in, number_of_channels_out):
    # two parameters:
    # (1) Multiply every element of the matrix by a parameter
    # (2) Take the average of all matrix elements, which gives a 1x1 matrix. Repeat that number to get a 12x15 matrix. Multiply the result by a parameter
    inp = layers.Reshape((5, number_of_channels_in))(inp)
    # ---(1)---
    out1 = layers.Conv1D(number_of_channels_out, 1, strides=1, padding='valid', use_bias=False, activation='relu')(inp)
    # ---(2)---
    out2 = layers.AveragePooling1D(pool_size=5, strides=1, padding='valid')(inp)
    repeated = [out2 for _ in range(5)]
    out2 = layers.Concatenate(axis=1)(repeated)
    out2 = layers.Conv1D(number_of_channels_out, 1, strides=1, padding='valid', use_bias=True, activation='relu')(out2)
    # return out1, out2
    return layers.Add()([out1,out2])

def get_deep_sets_network(pooling='sum'):
    number_of_channels = 30
    inp = layers.Input(shape=(5,))
    inp_list = [inp for _ in range(number_of_channels)]
    inp_duplicated = layers.Concatenate(axis=1)(inp_list)
    e1 = equivariant_layer(inp_duplicated, number_of_channels, number_of_channels)
    e1 = layers.Dropout(0.5)(e1)
    e2 = equivariant_layer(e1, number_of_channels, number_of_channels)
    e2 = layers.Dropout(0.5)(e2)
    e3 = equivariant_layer(e2, number_of_channels, number_of_channels)
    e3 = layers.Dropout(0.5)(e3)
  

    if pooling=='sum':
        p1 = layers.AveragePooling1D(5, strides=1, padding='valid')(e3)
    else:
        p1 = layers.MaxPooling1D(5, strides=1, padding='valid')(e3)
    p2 = layers.Flatten()(p1)
    fc1 = layers.Dense(64, activation='relu')(p2)
    fc2 = layers.Dense(32, activation='relu')(fc1)
    out = layers.Dense(1, activation='linear')(fc2)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model

def train_network(X_train, y_train, X_test, y_test):
    model = get_deep_sets_network()
    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    history = model.fit(
        X_train, y_train,
        epochs=999999,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    return model, history

def permute_vector(vector):
    return np.random.permutation(vector)

################################################################################
#running the program: 

if __name__ == '__main__':
    #training on the sasakain hodge numbers, as in the paper
    X,y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) #split data into training and testing
    print(get_deep_sets_network().summary()) #print an overview of the neural network created
    model, history = train_network(X_train, y_train, X_test, y_test) #train network on chosen data

    permuted_X_test = np.apply_along_axis(permute_vector, 1, X_test)
    print('Accuracy on ordered weights: ' + str(round(daattavya_accuracy(y_train, X_test, y_test, model)*100, 1)) + '%')
    print('Accuracy on randomly permuted weights: ' + str(round(daattavya_accuracy(y_train, permuted_X_test, y_test, model)*100, 1)) + '%')
    #expect these two accuracies to be similar if the NN is group invariant
