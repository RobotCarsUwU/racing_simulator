import tensorflow as tf
import keras
from keras import layers
import numpy as np

class MLP(keras.Model):
    def __init__(self, hidden_sizes, output_size, **kwargs):
        super(MLP, self).__init__(**kwargs)
        
        self.hidden_layers = []
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(layers.Dense(
                hidden_size, 
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(1e-4)
            ))
            self.hidden_layers.append(layers.Dropout(0.1))
        
        self.output_layer = layers.Dense(output_size, activation='tanh')
    
    def call(self, inputs, training=None):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.output_layer(x)
    
    def get_config(self):
        config = super(MLP, self).get_config()
        return config
