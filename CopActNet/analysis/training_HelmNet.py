#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:56:12 2021

@author: tenra
"""
import os
import sys
sys.path.insert(0, os.getcwd())
from CopActNet import config as cfg
from CopActNet.tools import reader
from sklearn.utils.class_weight import compute_sample_weight
from pathlib import Path
import pandas as pd
import numpy as np
from numpy.random import seed
from CopActNet.tools.visualizer import plotHelmNetHist
from CopActNet.models.helmNet import HelmNet, annotations_to_numbers

from tensorflow.keras.optimizers import Adam
from tensorflow.random import set_seed
NUM_EPOCHS = 1000

if __name__ == '__main__':
    # Loading the data
    X_train = reader.load_pickle(Path.joinpath(cfg.ROOT_DIR, "Annotations/Frames_forannotation.pkl"))
    X_active = reader.load_pickle(Path.joinpath(cfg.ROOT_DIR, "Annotations/FILE0024_annotated.pkl"))
    
    A_train = pd.read_csv(Path.joinpath(cfg.ROOT_DIR, "Annotations/labels_deeplearning-copenhagen_2021-11-09-02-43-16.csv"), header=None)
    A_active = pd.read_csv(Path.joinpath(cfg.ROOT_DIR, "Annotations/labels_file0024.csv"), header=None)
    
    X_test = reader.load_pickle(Path.joinpath(cfg.ROOT_DIR, "Annotations/FILE0020_annotated.pkl"))
    A_test = pd.read_csv(Path.joinpath(cfg.ROOT_DIR, "Annotations/labels_file0020.csv"), header=None)
    
    # Making the ys specific for the cyclists and helmets respectively
    cyclists_train, helmets_train = annotations_to_numbers(A_train, X_train.shape[0])
    cyclists_active, helmets_active = annotations_to_numbers(A_active, X_active.shape[0])
    cyclists_test, helmets_test = annotations_to_numbers(A_test, X_test.shape[0])
    
    # Removing a single observation with more than 2 cyclists
    rm_inds = [i for i, c in enumerate(cyclists_train) if c > 2]
    X_train = np.delete(X_train, rm_inds, axis=0)
    for ind in reversed(rm_inds):
        cyclists_train.pop(ind)
        helmets_train.pop(ind)
    
    # Make the ys in order to do sample weights
    y_train = []
    for c, h in zip(cyclists_train, helmets_train):
        y_train.append(cfg.HC_DICT[(c, h)])
    y_active = []
    for c, h in zip(cyclists_active, helmets_active):
        y_active.append(cfg.HC_DICT[(c, h)])
    
    # Changing the data types
    X_train, X_active, X_test = X_train.astype("float32"), X_active.astype("float32"), X_test.astype("float32")
    cyclists_train, helmets_train = np.array(cyclists_train, dtype="long"), np.array(helmets_train, dtype="long")
    cyclists_active, helmets_active = np.array(cyclists_active, dtype="long"), np.array(helmets_active, dtype="long")
    cyclists_test, helmets_test = np.array(cyclists_test, dtype="long"), np.array(helmets_test, dtype="long")
    
    
    # %% Training the base network
    set_seed(43)
    seed(43)
    
    net = HelmNet()
    
    # Compiling the network optimizer
    losses = {"cyclists" :  "sparse_categorical_crossentropy", "helmets" : "sparse_categorical_crossentropy"}
    loss_weights = {"cyclists": 1.0, "helmets": 1.0}
    optimizer = Adam(learning_rate=0.00001)
    
    net.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])
    
    # Sample weights computed based on balance in the y-samples (classes for all combinations of helmets and cyclists)
    hist = net.fit(x=X_train, y={"cyclists": cyclists_train, "helmets": helmets_train}, 
                   validation_data=(X_test, {"cyclists": cyclists_test, "helmets": helmets_test}), 
                   sample_weight=compute_sample_weight("balanced", y=y_train),
                   epochs=NUM_EPOCHS, batch_size=10, verbose=0)
    
    # Plotting training history
    plotHelmNetHist(hist, "base_model", cfg.ROOT_DIR)
    
    net.save_weights(Path.joinpath(cfg.HELMNET_WEIGHTS_PATH, "baseline.h5"))
    
    # Saving results for all models
    cyc_acc_act, cyc_acc_rand = [], []
    helm_acc_act, helm_acc_rand = [], []
    
    perc = [0]
    cyc_acc_act.append(hist.history['val_cyclists_accuracy'][-1])
    helm_acc_act.append(hist.history['val_helmets_accuracy'][-1])
    cyc_acc_rand.append(hist.history['val_cyclists_accuracy'][-1])
    helm_acc_rand.append(hist.history['val_helmets_accuracy'][-1])
    
    # %% Trying to use active learning to find most uncertain images
    preds = net.call(X_active)
    B = np.max((1 - np.max(preds["cyclists"], axis=1), 1 - np.max(preds["helmets"], axis=1)), axis=0)
    B_sorted = np.argsort(B)
    
    # Active learning loop
    for Np in [0.1, 0.15, 0.2, 0.25, 0.3]:
        # Taking the most uncertain Np percent from the active learning set
        N = int(X_active.shape[0] * Np)
        X_new = X_active[B_sorted[-N:]]
        cyclists_new = cyclists_active[B_sorted[-N:]]
        helmets_new = helmets_active[B_sorted[-N:]]
        y_new = np.array(y_active)[B_sorted[-N:]]
        
        set_seed(43)
        seed(43)
        net_active = HelmNet()
        net_active.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])
        
        # Sample weights computed based on the y_balance
        hist = net_active.fit(x=np.concatenate((X_train, X_new), axis=0), y={"cyclists": np.concatenate((cyclists_train, cyclists_new), axis=0), "helmets": np.concatenate((helmets_train, helmets_new), axis=0)}, 
                       validation_data=(X_test, {"cyclists": cyclists_test, "helmets": helmets_test}), 
                       sample_weight=compute_sample_weight("balanced", y=np.concatenate((y_train, y_new), axis=0)),
                       epochs=NUM_EPOCHS, batch_size=10, verbose=0)
        
        plotHelmNetHist(hist, f"{int(Np*100)}_percent_active", cfg.ROOT_DIR)
        
        perc.append(Np*100)
        cyc_acc_act.append(hist.history['val_cyclists_accuracy'][-1])
        helm_acc_act.append(hist.history['val_helmets_accuracy'][-1])
        # %% Using 10 percent at random
        chinds = np.random.choice(np.arange(0, len(X_active)), size=N, replace=False)
        X_new = X_active[chinds]
        cyclists_new = cyclists_active[chinds]
        helmets_new = helmets_active[chinds]
        y_new = np.array(y_active)[chinds]
        
        set_seed(43)
        seed(43)
        net_random = HelmNet()
        net_random.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])
        
        # Sample weights computed based on the y_balance
        hist = net_random.fit(x=np.concatenate((X_train, X_new), axis=0), y={"cyclists": np.concatenate((cyclists_train, cyclists_new), axis=0), "helmets": np.concatenate((helmets_train, helmets_new), axis=0)}, 
                       validation_data=(X_test, {"cyclists": cyclists_test, "helmets": helmets_test}), 
                       sample_weight=compute_sample_weight("balanced", y=np.concatenate((y_train, y_new), axis=0)),
                       epochs=NUM_EPOCHS, batch_size=10, verbose=0)
        
        plotHelmNetHist(hist, f"{int(Np*100)}_percent_random", cfg.ROOT_DIR)
        
        cyc_acc_rand.append(hist.history['val_cyclists_accuracy'][-1])
        helm_acc_rand.append(hist.history['val_helmets_accuracy'][-1])
    
    
    # %% Training on all the data available
    set_seed(43)
    seed(43)
    netall = HelmNet()
    netall.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=["accuracy"])
    
    # Sample weights computed based on the y_balance
    hist = netall.fit(x=np.concatenate((X_train, X_active), axis=0), y={"cyclists": np.concatenate((cyclists_train, cyclists_active), axis=0), "helmets": np.concatenate((helmets_train, helmets_active), axis=0)}, 
                   validation_data=(X_test, {"cyclists": cyclists_test, "helmets": helmets_test}), 
                   sample_weight=compute_sample_weight("balanced", y=np.concatenate((y_train, y_active), axis=0)),
                   epochs=NUM_EPOCHS, batch_size=10, verbose=1)
    
    plotHelmNetHist(hist, "Using all data", cfg.ROOT_DIR)
    perc.append(100)
    cyc_acc_act.append(hist.history['val_cyclists_accuracy'][-1])
    helm_acc_act.append(hist.history['val_helmets_accuracy'][-1])
    cyc_acc_rand.append(hist.history['val_cyclists_accuracy'][-1])
    helm_acc_rand.append(hist.history['val_helmets_accuracy'][-1])
    
    netall.save_weights(Path.joinpath(cfg.HELMNET_WEIGHTS_PATH, "final.h5"))
    
    # %% Writing the results to a file
    file = open(cfg.HELMNET_RESULTS_PATH, "w")
    file.write(f"{'Percentage'},{'Cyclists act.'},{'Cyclists rand.'},{'Helmets act.'},{'Helmets rand.'}\n")
    for p, c_act, c_rand, h_act, h_rand in zip(perc, cyc_acc_act, cyc_acc_rand, helm_acc_act, helm_acc_rand):
        file.write(f"{int(p): ^10d},{c_act: ^13.4f},{c_rand: ^14.4f},{h_act: ^12.4f},{h_rand: ^13.4f}\n")
    file.close()
