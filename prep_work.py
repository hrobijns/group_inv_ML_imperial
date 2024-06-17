#Original file is located at: https://colab.research.google.com/drive/1AUofaupklsbrcfQlN5HKs8n2Rvwaq9wF

import requests
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request


def data_wrangle(data_string):
#loads in and cleans up the data used for the neural network
    file_path = f'{data_string}.txt'
    file_url = f'https://raw.githubusercontent.com/TomasSilva/MLcCY7/main/Data/{data_string}.txt'
    try:
        with open(file_path, 'r') as file:
            data_raw = file.read()
    except FileNotFoundError as e:
        urllib.request.urlretrieve(file_url, file_path)
        with open(file_path, 'r') as file:
            data_raw = file.read()
    data_list = ast.literal_eval(data_raw) #converts from string to list
    data_array = np.array(data_list) #converts from list to NumPy array
    return data_array
    
def get_network():
#define the neural network
    inp = tf.keras.layers.Input(shape=(5,))
    prep = tf.keras.layers.Flatten()(inp)
    h1 = tf.keras.layers.Dense(1000, activation=tf.nn.relu)(prep)
    h2 = tf.keras.layers.Dense(100, activation=tf.nn.relu)(h1)
    out = tf.keras.layers.Dense(2, activation=tf.nn.relu)(h2)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics = ['accuracy']
    )
    return model

def train_network(X_train, y_train, X_test, y_test):
#train the network
    model = get_network()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(
        X_train, y_train,
        epochs=999999,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    return history

if __name__ == '__main__':
#run the program
    X = data_wrangle('WP4s')
    y = data_wrangle('WP4_Hodges')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) #split data into training and testing
    X_train, X_test = tf.keras.utils.normalize(X_train, axis=1), tf.keras.utils.normalize(X_test, axis=1) #normalise
