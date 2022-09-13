#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various useful functions.
"""

import matplotlib.pyplot as plt


def plot_loss(history, title='model loss'):
    '''
    Plots the loss of the training against the validation set.
    '''
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_accuracy(history, title='model accuracy'):
    '''
    Plots the accuracy of the training against the validation set.
    '''
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()