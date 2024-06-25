#Original file is located at: https://colab.research.google.com/drive/1AUofaupklsbrcfQlN5HKs8n2Rvwaq9wF

import requests
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request

################################################################################
#importing and wrangling data

def data_wrangle_CY(data_string):
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

def data_wrangle_sasakian(data_string):
    Sweights, SHodge = [], []
    try:
        with open('Topological_Data.txt','r') as file:
            for idx, line in enumerate(file.readlines()[1:]):
                if idx%6 == 0: Sweights.append(eval(line))
                if idx%6 == 2: SHodge.append(eval(line))
    except FileNotFoundError as e:
        urllib.request.urlretrieve('https://raw.githubusercontent.com/TomasSilva/MLcCY7/main/Data/Topological_Data.txt', 'Topological_Data.txt')
        with open('Topological_Data.txt','r') as file:
            for idx, line in enumerate(file.readlines()[1:]):
                if idx%6 == 0: Sweights.append(eval(line))
                if idx%6 == 2: SHodge.append(eval(line))
    return Sweights, SHodge

################################################################################
#defining and training NN

def get_network():
    inp = tf.keras.layers.Input(shape=(5,))
    prep = tf.keras.layers.Flatten()(inp)
    h1 = tf.keras.layers.Dense(16, activation='relu')(prep)
    h2 = tf.keras.layers.Dense(32, activation='relu')(h1)
    h3 = tf.keras.layers.Dense(16, activation='relu')(h2)
    out = tf.keras.layers.Dense(2, activation='linear')(h3)

    model = tf.keras.models.Model(inputs=inp, outputs=out)
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics = ['accuracy']
    )
    return model

def train_network(X_train, y_train, X_test, y_test):
    model = get_network()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(
        X_train, y_train,
        epochs=999999,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    return history

################################################################################
#run program: this example trains on the CY numbers, but using data_wrangle_sasakian() we can also train on the saskain numbers, as the paper does. 

if __name__ == '__main__':
    X = data_wrangle('WP4s')
    y = data_wrangle('WP4_Hodges')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) #split data into training and testing

    model = get_network()    
    print(model.summary()) #print an overview of the neural network created
    history = train_network(X_train, y_train, X_test, y_test) #train network on chosen data
