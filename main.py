
# coding: utf-8

# In[1]:

import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression
import os
import numpy as np


# In[2]:

class agent():
    def __init__(self, name, feature, label):
        self.name = name
        self.feature = np.array(feature)
        self.label = np.array(label)
        
        self.feature_size = self.feature.shape
        self.feature_size = self.feature_size[1]
        self.label_size = self.label.shape
        self.label_size = self.label_size[1]
        self.model = self.create_model()
        
        self.main_dir = self.name
        self.model_dir = self.main_dir + '/model_save'
        
        if not os.path.exists(self.main_dir):
            os.makedirs(self.main_dir)
            
    def saver(self):
        self.model.save(self.model_dir)
    
    def restore(self):
        self.model.load(self.model_dir)
        
    def train_for(self, epoch):
        self.model.fit({'input': self.feature}, {'targets': self.label}, n_epoch=epoch, snapshot_step=500, show_metric=True, run_id='openai_learning')
    
    def create_model(self):
        keep = 0.8
        LR = 1e-3
        network = input_data(shape=[None, self.feature_size], name='input')
        network = fully_connected(network, 128, activation='relu', name = 'hidden_1')
        network = dropout(network,keep)
        network = fully_connected(network, 256, activation='relu', name = 'hidden_2')
        network = dropout(network,keep)
        network = fully_connected(network, 512, activation='relu', name = 'hidden_3')
        network = dropout(network,keep)
        network = fully_connected(network, 256, activation='relu', name = 'hidden_4')
        network = dropout(network,keep)
        network = fully_connected(network, 128, activation='relu', name = 'hidden_5')
        network = dropout(network,keep)
        network = fully_connected(network, self.label_size, activation='softmax', name = 'softmax')
        network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
        model = tflearn.DNN(network, tensorboard_verbose=3)
        return model
        

