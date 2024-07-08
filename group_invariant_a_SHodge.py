from vanilla_SHodge import data_wrangle_S
from vanilla_SHodge import daattavya_accuracy
from vanilla_SHodge import train_network
from group_invariant_b_SHodge import permute_vector
from group_invariant_b_SHodge import permutation_invariance_confirmation

import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

################################################################################
# define architecture of NN according to the second-half of section 3.1 of the paper

class ExpandDimsLayer(layers.Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=1)

class TileLayer(layers.Layer):
    def __init__(self, multiples, **kwargs):
        super().__init__(**kwargs)
        self.multiples = multiples
    
    def call(self, inputs):
        return tf.tile(inputs, self.multiples)

def equivariant_layer(inp, number_of_channels_in, number_of_channels_out):
    # introduce first parameter
    out1 = layers.Conv1D(number_of_channels_out, 1, strides=1, padding='valid', use_bias=False, activation='relu')(inp)
    
    # pooling function
    out2 = layers.GlobalAveragePooling1D()(inp)
    
    # adjust shapes and introduce second parameter
    out2 = ExpandDimsLayer()(out2)
    out2 = layers.Conv1D(number_of_channels_out, 1, strides=1, padding='valid', use_bias=True, activation='relu')(out2)
    out2 = TileLayer(multiples=[1, inp.shape[1], 1])(out2)
    
    return layers.Add()([out1, out2])

def get_network():
    number_of_channels = 100
    inp = layers.Input(shape=(5, 1)) 
   
    # apply equivariant layers
    e1 = equivariant_layer(inp, 1, number_of_channels)
    e1 = layers.Dropout(0.5)(e1)
  
    e2 = equivariant_layer(e1, number_of_channels, number_of_channels)
    e2 = layers.Dropout(0.5)(e2)
    
    # pooling function
    p1 = layers.GlobalAveragePooling1D()(e2)

    # further training
    fc1 = layers.Dense(16, activation='relu')(p1)
    fc2 = layers.Dense(32, activation='relu')(fc1)
    fc3 = layers.Dense(16, activation='relu')(fc2)

    out = layers.Dense(1, activation='linear')(fc3)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer = tf.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model

################################################################################
# running the program: 

if __name__ == '__main__':
    X, y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) # split data into training and testing
    model, history = train_network(X_train, y_train, X_test, y_test, get_network()) # train network on chosen data
    print('Accuracy: ' + str(round(daattavya_accuracy(y_train, X_test, y_test, model) * 100, 1)) + '%')
    permutation_invariance_confirmation(model, X_test)
