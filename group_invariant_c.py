from vanilla_SHodge import data_wrangle_S
from vanilla_SHodge import daattavya_accuracy
from vanilla_SHodge import train_network
from group_invariant_b import permute_vector
from group_invariant_b import permutation_invariance_confirmation

import numpy as np
import requests
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request
import itertools

################################################################################
# define architecture of NN

def get_network():
    inp = tf.keras.layers.Input(shape=(5,))
    
    # generate all permutations of the input vector
    permuted_inputs = tf.keras.layers.Lambda(lambda x: [tf.gather(x, indices=perm, axis=1) 
                                                        for perm in itertools.permutations(range(5))])(inp)
    
    # define shared model which runs in parallel for all permutations
    def get_shared_parallel_model():
        inp = tf.keras.layers.Input(shape=(5,))
        prep = tf.keras.layers.Flatten()(inp)
        h1 = tf.keras.layers.Dense(16, activation='relu')(prep)
        h2 = tf.keras.layers.Dense(32, activation='relu')(h1)
        h3 = tf.keras.layers.Dense(16, activation='relu')(h2)
        out = tf.keras.layers.Dense(1, activation='linear')(h3)
        return tf.keras.models.Model(inp, out)
    
    shared_model = get_shared_parallel_model()
    
    # apply the shared model to each permuted input
    parallel_outputs = [shared_model(perm) for perm in permuted_inputs]
    
    # sum the outputs of these individual models running in parallel
    parallel_outputs_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.stack(x), axis=0))(parallel_outputs)
    
    # define the overall model
    model = tf.keras.models.Model(inputs=inp, outputs=parallel_outputs_sum)
    
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )
    
    return model

################################################################################
# running the program: 

if __name__ == '__main__':
    X,y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) # split data into training and testing
    model, history = train_network(X_train, y_train, X_test, y_test, get_network()) # train network on chosen data
    print('Accuracy: ' + str(round(daattavya_accuracy(y_train, X_test, y_test, model)*100, 1)) + '%')
    permutation_invariance_confirmation(model, X_test)
    