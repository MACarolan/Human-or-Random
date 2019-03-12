# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:11:47 2018

@author: Michael
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random


def train(length, n_ep, show=False):
    """
    Train over the length dataset over n_ep epochs, contructing a graph of loss
    over time if show.
    
    length: integer representing string length to train over
    
    n_ep: integer representing the number of epochs to train over
    
    show: boolean indicating whether to graph loss and accuracy or not
    """
    al_dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 
               'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 
               'r':17, 's':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 
               'z':25}
    
    #Load proper data
    raw_rand = open(f'Length {length} Data/rand_data_len{length}.txt', 'r').readlines()
    raw_human = open(f'Length {length} Data/human_data_len{length}.txt','r').readlines()
    
    #Turn data into numbers
    raw_rand = [string[:length] for string in raw_rand]
    raw_human = [string[:length] for string in raw_human]
    
    raw_rand = [list(string) for string in raw_rand]
    raw_human = [list(string) for string in raw_human]
    
    for lists in raw_rand:
        for i in range(length):
            lists[i] = al_dict[lists[i]]
    for lists in raw_human:
        for i in range(length):
            lists[i] = al_dict[lists[i]]
    
    #Set aside 1/3 for test data
    test_rand = raw_rand[:len(raw_rand)//3]
    test_human = raw_human[:len(raw_human)//3]
    train_rand = raw_rand[len(raw_rand)//3:]
    train_human = raw_human[len(raw_human)//3:]
    
    #Intersperse both samples randomly with labels (0 = rand, 1 = human)
    train = []
    trn_labels = []
    test = []
    tst_labels = []
    while len(test_rand) and len(test_human) != 0:
        num = random.random()
        if num > .5 and len(test_rand) > 0:
            test.append(test_rand.pop())
            tst_labels.append(0)
        else:
            test.append(test_human.pop())
            tst_labels.append(1)
            
    while len(train_rand) and len(train_human) != 0:
        num = random.random()
        if num > .5 and len(train_rand) > 0:
            train.append(train_rand.pop())
            trn_labels.append(0)
        else:
            train.append(train_human.pop())
            trn_labels.append(1)
            
    # Build network, 2 hidden layers, 26 dimensional vectors for alphabet
    model = keras.Sequential()
    model.add(keras.layers.Embedding(26, 16)) #26=numletters
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    
    # Loss function, Using Probabilities
    model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    # Perform n_ep epochs of training on train_data
    partial_train = np.array(train[len(train)//10:])
    partial_labels = trn_labels[len(train)//10:]
    val_train = np.array(train[:len(train)//10])
    val_labels = trn_labels[:len(train)//10]
    print((len(partial_train), len(partial_labels)),
          (len(val_train), len(val_labels)),
          (len(test), len(tst_labels)))
    
    history = model.fit(partial_train,
                        partial_labels,
                        epochs=n_ep,
                        batch_size=512,
                        validation_data=(val_train, val_labels),
                        verbose=1)
    
    #EVALUATE THE FINAL MODEL

    results = model.evaluate(np.array(test), tst_labels)
    print(results)
    
    # GRAPH ACCURACY AND LOSS OVER TIME
    if show:
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        history_dict = history.history
        
        epochs = range(1, len(acc) + 1)
        
        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.show()
        
        plt.clf()   # clear figure
        acc_values = history_dict['acc']
        val_acc_values = history_dict['val_acc']
        
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.show()
        
    return model
        
def string_to_list(string):
    """
    Converts a string into a list of numbers representing letters
    """
    al_dict = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8, 
               'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 
               'r':17, 's':18, 't':19, 'u':20, 'v':21, 'w':22, 'x':23, 'y':24, 
               'z':25}
    
    out = list(string)
    
    for i in range(len(string)):
            out[i] = al_dict[out[i]]
    
    return out
    