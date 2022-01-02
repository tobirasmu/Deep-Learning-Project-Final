#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 14:26:02 2021

@author: tenra
"""
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPool2D, BatchNormalization, Flatten
import pandas as pd


class HelmNet(Model):
    
    def __init__(self):
        super(HelmNet, self).__init__()
        
        self.c1 = Conv2D(8, 5, padding='same', activation="relu")
        self.pool1 = MaxPool2D(padding='same')
        self.BN1 = BatchNormalization()
        
        self.c2 = Conv2D(16, 3, padding='same', activation="relu")
        self.pool2 = MaxPool2D(padding='same')
        self.BN2 = BatchNormalization()
        
        self.C3 = Conv2D(16, 3, padding='same', activation='relu')
        self.BN3 = BatchNormalization()
        
        self.L1 = Dense(512, activation='relu')
        self.BN4 = BatchNormalization()
        
        self.L2 = Dense(512, activation='relu')
        self.BN5 = BatchNormalization()
        
        self.L3c = Dense(256, activation='relu')
        self.L3h = Dense(256, activation='relu')
        
        self.lOcyclists = Dense(3, activation='softmax')
        self.lOhelmets = Dense(3, activation='softmax')
        
        self.dropout = Dropout(0.15)
        self.dropout2 = Dropout(0.5)
        
        
    def call(self, x):
        
        x = self.c1(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.BN1(x)
        
        x = self.c2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.BN2(x)
        
        x = self.C3(x)
        x = self.dropout(x)
        x = self.BN3(x)
        # Flattening 
        x = Flatten()(x)
        
        x = self.L1(x)
        x = self.dropout(x)
        x = self.BN4(x)
        
        x = self.L2(x)
        x = self.dropout(x)
        x = self.BN5(x)
        
        xc = self.dropout(self.L3c(x))
        xh = self.dropout2(self.L3h(x))
        
        return {"cyclists" : self.lOcyclists(xc), "helmets" : self.lOhelmets(xh)}


def annotations_to_numbers(annotations: pd.DataFrame, num_frames: int):
    annotations.columns = ("A", "B", "C", "D", "E", "F", "G", "H")
    
    cyclists, helmets = [], []

    for frame in range(num_frames):
        frame_name = f"FRAME{frame:0>5d}.jpg"
        subset = annotations.A[annotations.F == frame_name]
        c, h = 0, 0
        for obj in subset:

            if obj == "Cyclist" or obj == "Cyclists":

                c += 1
            else:
                h += 1
        cyclists.append(c)
        helmets.append(h)
    return cyclists, helmets