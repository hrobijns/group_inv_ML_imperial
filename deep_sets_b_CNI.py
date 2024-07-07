# import necessary libraries but also useful previously defined functions
from CNI_learn import data_wrangle_CNI
from CNI_learn import classification_accuracy
from deep_sets_b_SHodge import permute_vector
from deep_sets_b_SHodge import permutation_invariance_confirmation
from SHodge_learn import train_network

import requests
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request

################################################################################
# re-define deep sets network as a classifier

def get_deep_sets_classifier():
    inp = tf.keras.layers.Input(shape=(5,))
    
    # split the input vector into 5 elements
    splits = tf.keras.layers.Lambda(lambda x: [x[:, i:i+1] for i in range(5)])(inp)
    
    # define shared model which runs in parallel for all 5 elements
    def get_shared_parallel_model():
        input_element = tf.keras.layers.Input(shape=(1,))
        h_a = tf.keras.layers.Dense(32, activation='relu')(input_element)
        h_b = tf.keras.layers.Dense(64, activation='relu')(h_a)
        output_element = tf.keras.layers.Dense(128, activation='relu')(h_b)
        return tf.keras.models.Model(input_element, output_element)
    
    shared_model = get_shared_parallel_model()
    
    # apply the shared model to each element
    parallel_outputs = [shared_model(split) for split in splits]
    
    # sum the outputs of these individual models running in parallel
    parallel_outputs_sum = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=0))(parallel_outputs)
  
    # further training on the summed output
    h1 = tf.keras.layers.Dense(64, activation='relu')(parallel_outputs_sum)
    h2 = tf.keras.layers.Dense(32, activation='relu')(h1)
    final_output = tf.keras.layers.Dense(24, activation='softmax')(h2)
    
    # define the overall model
    model = tf.keras.models.Model(inputs=inp, outputs=final_output)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics = ['accuracy']
    )
    return model 

################################################################################
# run the program:
if __name__ == '__main__':
    X,y = data_wrangle_CNI()
    y = (y-1)/2
    y = np.round(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) # split data into training and testing
    model, history = train_network(X_train, y_train, X_test, y_test, get_deep_sets_classifier()) # train network on chosen data
    print('Accuracy: ' + str(round(classification_accuracy(X_test, y_test, model)*100, 1)) + '%')
    permutation_invariance_confirmation(model, X_test)

