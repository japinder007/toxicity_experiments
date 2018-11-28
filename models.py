import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['KERAS_BACKEND']='tensorflow'
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.models import Model,Sequential

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, optimizers


def lstm_keras(inp_dim, vocab_size, embed_size, num_classes, learn_rate):
#     K.clear_session()
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True))
    model.add(Dropout(0.25))
    model.add(LSTM(embed_size))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    print model.summary()
    return model


def blstm(inp_dim,vocab_size, embed_size, num_classes, learn_rate):   
#     K.clear_session()
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=inp_dim, trainable=True))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(embed_size)))
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model


class AttLayer(Layer):

    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1],),
                                      initializer='random_normal',
                                      trainable=True)
        super(AttLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def blstm_atten(inp_dim, vocab_size, embed_size, num_classes, learn_rate):
#     K.clear_session()
    model = Sequential()
    model.add(Embedding(vocab_size, embed_size, input_length=inp_dim))
    model.add(Dropout(0.25))
    model.add(Bidirectional(LSTM(embed_size, return_sequences=True)))
    model.add(AttLayer())
    model.add(Dropout(0.50))
    model.add(Dense(num_classes, activation='softmax'))
    adam = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()
    return model

def get_model(m_type,inp_dim, vocab_size, embed_size, num_classes, learn_rate):
    if m_type == 'lstm':
        model = lstm_keras(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    elif m_type == "blstm":
        model = blstm(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    elif m_type == "blstm_attention":
        model = blstm_atten(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    else:
        print "ERROR: Please specify a correst model"
        return None
    return model