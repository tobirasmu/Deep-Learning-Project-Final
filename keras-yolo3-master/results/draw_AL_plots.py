# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Read AL predictions
al = pd.read_csv('AL.txt')


x_vals = np.array([10, 15, 20, 25, 30])
fig, ax = plt.subplots(1, 1)
ax.set_facecolor((212/255, 208/255, 200/255))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("% of labeled images")
ax.set_ylabel("Average Precision")
ax.grid()
ax.set_xticks(x_vals)

cyc_active_vals = al.iloc[1:6,1]
helm_active_vals = al.iloc[1:6,2]
hovd_active_vals = al.iloc[1:6,3]

cyc_rand_vals = al.iloc[7:12,1]
helm_rand_vals = al.iloc[7:12,2]
hovd_rand_vals = al.iloc[7:12,3]



ax.plot(x_vals,cyc_rand_vals, 'b-x', label="Cyclists Random")
ax.plot(x_vals,cyc_active_vals, 'r-x', label="Cyclists Active")
ax.plot(x_vals, helm_rand_vals, 'b--x', label="Helmets Random")
ax.plot(x_vals, helm_active_vals, 'r--x', label="Helmets Active")
ax.plot(x_vals, hovd_rand_vals, 'b:x', label="Hövding Random")
ax.plot(x_vals, hovd_active_vals, 'r:x', label="Hövding Active")

ax.legend(prop={'size': 9})
fig.show()

x_vals = np.array([10, 15, 20, 25, 30])
fig, ax = plt.subplots(1, 1)
ax.set_facecolor((212/255, 208/255, 200/255))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("% of labeled images")
ax.set_ylabel("Mean Average Precision")
ax.grid()
ax.set_xticks(x_vals)

active_map = al.iloc[1:6,4]
random_map = al.iloc[7:12,4]

ax.plot(x_vals,random_map, 'b-x', label="Random")
ax.plot(x_vals,active_map, 'r-x', label="Active")

ax.legend()
fig.show()
