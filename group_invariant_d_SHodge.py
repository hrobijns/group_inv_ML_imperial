from vanilla_SHodge import data_wrangle_S
from vanilla_SHodge import get_network
from vanilla_SHodge import train_network

import requests
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import urllib.request
import itertools

################################################################################
# defining a new accuracy which is group invariant: 

def group_invariant_accuracy(training_outputs, test_inputs, test_outputs, model):
    bound = 0.05 * (np.max(training_outputs) - np.min(training_outputs))  # define the bound as in Daattavya's paper
    all_permutations = list(itertools.permutations(test_inputs))  # generate all permutations of the input vector
    predictions = [model.predict(np.array(perm).reshape(1, -1))[0] for perm in all_permutations]  # predict for each permutation and reshape to match model input
    averaged_prediction = np.mean(predictions)
    accuracy = np.mean(np.where(np.abs(averaged_prediction - test_outputs) < bound, 1, 0))
    return accuracy

################################################################################
# running the program: 

if __name__ == '__main__':
    # training on the sasakain hodge numbers, as in the paper
    X,y = data_wrangle_S()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5) # split data into training and testing
    model, history = train_network(X_train, y_train, X_test, y_test, get_network()) # train network on chosen data
    print('Accuracy as defined in the paper: ')
    print(str(round(group_invariant_accuracy(y_train, X_test, y_test, model)*100, 1)) + '%')
