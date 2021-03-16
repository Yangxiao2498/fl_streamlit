#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:56:43 2021

@author: john.onwuemeka
"""


import numpy as np
import tensorflow as tf


from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from numpy import newaxis
from nmt_utils import *
import pickle


# Define global variables
n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64 # number of units for the post-attention LSTM's hidden state "s"
Tx = 30 # max length of the input text
Ty = 10 # lenth of the output text
m = 1 # no. of inputs in each item of the input list

# Defined shared layers as global variables
Tx = 30 #max length of input text
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """
    
    # repeator    
    s_prev = repeator(s_prev)
    
    # Use concatenator to concatenate a and s_prev on the last axis
    concat = concatenator([a,s_prev])
    
    # compute the "intermediate energies" variable e.
    e = densor1(concat)
   
    #compute the "energies" variable energies.
    energies = densor2(e)
    
    #compute the attention weights "alphas"
    alphas = activator(energies)
    
    #compute the context vector to be given to the next (post-attention) LSTM-cell
    context = dotor([alphas,a])
    
    return context

def kmodel(Tx, Ty, n_a, n_s, hv,post_activation_LSTM_cell,output_layer):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    # Define the inputs of your model, as well as s0 (initial hidden state) 
    # and c0 (initial cell state) for the decoder LSTM
    X = Input(shape=(Tx, hv))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    
    
    # Define your pre-attention Bi-LSTM.
    a = Bidirectional(LSTM(units=n_a, return_sequences=True))(X)
    
    # Iterate for Ty steps
    for t in range(Ty):
    
        # get the context vector at step t
        context = one_step_attention(a,s)
        
        # Apply the post-attention LSTM cell to the "context" vector.
        s, _, c = post_activation_LSTM_cell(inputs=context, initial_state=[s,c])
        
        # Apply Dense layer to the hidden state output of the post-attention LSTM
        out = output_layer(s)
        
        # Append the outputs to create an output list
        outputs.append(out)
    
    # Create model instance
    model = Model(inputs=[X,s0,c0],outputs=outputs)

    return model


def model_arch(n_a,n_s,Tx,Ty,hv,mv):
    """
    Build and compile the model instance (architecture)

    Parameters
    ----------
    n_a : hidden state size of the Bi-LSTM
    n_s : hidden state size of the post-attention LSTM
    Tx : max length of input text
    Ty : length of output text
    hv : size of the human vocabulary dictionary
    mv : size of the machine vocabulary dictionary

    Returns
    -------
    model : model instance (architecture)

    """   

    post_activation_LSTM_cell = LSTM(n_s, return_state = True) # post-attention LSTM 
    output_layer = Dense(mv, activation=softmax)
    
    # build model
    model = kmodel(Tx, Ty, n_a, n_s, hv,post_activation_LSTM_cell,output_layer)

    return model


def date_format(datein):
    """
    Function thst uses the trained RNN weights to predict dates in YYYY-MM-DD
    format given a date input in human readable format 

    Parameters
    ----------
    datein : Input date in human readable format (e.g., Tue 10 April 2013)

    Returns
    -------
    dateout : Output date in YYYY-MM-DD format

    """
    
    datein = 'Tuesday 10 Jul 2007'
    # load the human vocabulary dictionary
    pfile = open('./date_model/human_vocab.pickle','rb')
    human_vocab = pickle.load(pfile)
    pfile.close()
    
    # load the inverted machine vocabulary dictionary
    pfile = open('./date_model/inv_machine_vocab.pickle','rb')
    inv_machine_vocab = pickle.load(pfile)
    pfile.close()
    

    # load the machine vocabulary dictionary
    pfile = open('./date_model/machine_vocab.pickle','rb')
    machine_vocab = pickle.load(pfile)
    pfile.close()
    
    hv = len(human_vocab)
    mv = len(machine_vocab)
    
    # build model architecture
    model = model_arch(n_a,n_s,Tx,Ty,hv,mv)
    
    # load pre-trained weights
    model.load_weights('./date_model/trained_date_model.h5')
    
    #initialize hidden state weights
    s0 = np.zeros((m, n_s))
    c0 = np.zeros((m, n_s))
    
    #convert date text to integers    
    datein_int= string_to_int(datein, Tx, human_vocab)
    
    #format date_int to model input structure 
    datein_int = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), datein_int)))
    datein_int = datein_int[newaxis,:,:]
    
    # Predict the output
    pred = model.predict([datein_int, s0, c0])
    pred = np.argmax(pred, axis = -1)
    dateout = [inv_machine_vocab[int(i)] for i in pred]
    dateout = ''.join(dateout)
    
    return dateout


if __name__ == "__main__":
    
    #get user input
    datein = input("please enter date in human readable format, e.g. 3 May 1979,Tue 10 Jul 2007: ")
    
    #get run model and get the formatted date
    dateout = date_format(datein)
    
    print(dateout)
    
