# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:35:45 2021

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate
import h5py

font = {'family' : 'serif',
        'size'   : 6}

mpl.rc('font', **font)


# %% 


fig = plt.figure(figsize=(3.375, 3.375), dpi=300)
ax = plt.subplot()

x = np.linspace(-np.pi, np.pi, num=2048, endpoint=True)
v = np.linspace(-np.pi, np.pi, num=2048, endpoint=True)

xg, vg = np.meshgrid(x, v)

ham = vg**2/2 - np.cos(xg)

plt.contour(x, v, ham, levels=13)

plt.xlabel('$x$')
plt.ylabel('$p$')

plt.tight_layout(h_pad=0.6)
plt.tight_layout(h_pad=0.6)