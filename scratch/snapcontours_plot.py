# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:47:18 2021

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate
import h5py

# %%

case = 2

cdata = np.load('case{}_snapcontours.npz'.format(case))
data = np.load('case{}_mixing_lengths_1.npz'.format(case))
qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))

simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, 3-case), 'r')

index = 0
q = simdata['tasks/q'][index,:,:]

def circularInterpolant(x, vec, ofs, pad):
    xp = np.zeros(len(x)+2*pad)
    xp[pad:-pad] = x
    xp[:pad] = x[-pad:] - ofs
    xp[-pad:] = x[:pad] + ofs
    
    
    vecp = np.zeros(len(x)+2*pad)
    vecp[pad:-pad] = vec
    vecp[:pad] = vec[-pad:]
    vecp[-pad:] = vec[:pad]

    return scipy.interpolate.interp1d(xp, vecp, kind='quadratic')


nx = 2048
x = np.linspace(-np.pi,np.pi, num=nx, endpoint=False)


# %%

plt.figure()
ax = plt.subplot(111)
ax.plot(np.average(cdata['lenmaxcontour'], axis=0))

axt = ax.twinx()
axt.plot(np.average(data['allcorrdims'], axis=1), c='tab:orange')

kuofraction = (np.sum(np.gradient(qbars['qbar'], axis=1)/(2*np.pi/nx)<-8, axis=0)) / qbars['qbar'].shape[0]
axt.plot(np.arange(nx)/nx*521, kuofraction+1, c='tab:green')

# %%


mq = circularInterpolant(cdata['levels'], np.average(cdata['lenmaxcontour'], axis=0), 2*8*np.pi, 500)

plt.figure()
plt.imshow(mq(q+8*x[:,np.newaxis]), origin='lower')

plt.colorbar()