import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD
from NeuralNetwork_util import f1_score


# Funcao para criar um modelo conforme requerido pelo KerasClassifier
def baseline_mlp(number_of_features=9, neurons=1,optimizer='adam',activation='relu'):
    # create model
    # #
    #
    model = Sequential()
    model.add(Dense(neurons, input_dim=number_of_features, kernel_initializer='normal', activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy','mse'])
    return model

def baseline_sgd(number_of_features=9, neurons = 1, learn_rate=0.01, momentum=0, activation = 'relu'):
    model = Sequential()
    model.add(Dense(neurons, input_dim=number_of_features, kernel_initializer='normal', activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy','mse'])
    return model

def baseline_deep(number_of_features=9, neurons1 = 1,neurons2 = 1,neurons3 = 1, optimizer='adam', activation = 'relu'):
    # create model
    # #
    #
    model = Sequential()
    model.add(Dense(neurons1, input_dim=number_of_features, kernel_initializer='normal', activation = activation))
    model.add(Dense(neurons2, kernel_initializer='normal', activation=activation))
    model.add(Dense(neurons3, kernel_initializer='normal', activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', metrics=['accuracy','mse'], optimizer=optimizer)
    return model

def baseline_dropout(number_of_features=9, neurons1 = 1,neurons2 = 1,neurons3 = 1, dropout_rate=0.0):
    # create model
    # #
    #
    model = Sequential()
    model.add(Dense(neurons1, input_dim=number_of_features, kernel_initializer='normal', activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons2, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons3, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', metrics=['accuracy','mse'], optimizer= 'adam')
    return model

def baseline_dropout_input(number_of_features=9,neurons1 = 1,neurons2 = 1,neurons3 = 1, dropout_rate=0.0):
    # create model
    # #
    #
    model = Sequential()
    model.add(Dropout(dropout_rate, input_shape=(number_of_features,)))
    model.add(Dense(neurons1, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons2, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons3, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', metrics=['accuracy','mse'], optimizer= 'adam')
    return model