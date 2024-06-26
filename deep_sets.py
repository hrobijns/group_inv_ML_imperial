#import necessary libraries but also useful previously defined functions 
from prep_work import data_wrangle_S
from prep_work import train_network
from prep_work import daattavya_accuracy

import requests
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request

################################################################################
#define new architecture for the NN, same as in prep_work.py but with an additional, "summed" layer
def get_network_deep_sets():
    inp = tf.keras.layers.Input(shape=(5,))
    prep = tf.keras.layers.Reshape((5,))(inp)
    h1 = tf.keras.layers.Dense(16, activation='relu')(prep)
    h2 = tf.keras.layers.Dense(32, activation='relu')(h1)
    #adding a layer to the network which sums the inputs from the previous layer
    summed = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(h2)
    out = tf.keras.layers.Dense(1, activation='linear')(summed)
    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001)
    )
    return model

################################################################################
#running the program: 

if __name__ == '__main__':
    #training on the sasakain hodge numbers, as in the paper
    X,y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) #split data into training and testing
    print(get_network_deep_sets().summary()) #print an overview of the neural network created
    model, history = train_network(X_train, y_train, X_test, y_test) #train network on chosen data
    print('Accuracy as defined in the paper: ')
    print(str(round(daattavya_accuracy(X, y, model)*100, 1)) + '%')
