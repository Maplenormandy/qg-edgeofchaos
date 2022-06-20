# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:55:40 2022

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

# %%

case = 2
simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, 3-case), 'r')

# %%

index = 0
q = simdata['tasks/q'][index,:,:]
qfft = np.fft.rfft2(q)

nx = 2048
x = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)

colors = plt.cm.twilight(np.linspace(0,1,32, endpoint=False))

numzones = 3
colors2 = np.vstack(list([mpl.cm.twilight.colors for i in range(numzones+2)]))
qbrange = 8.0*(2*np.pi/numzones)*(numzones+2)
mymap2 = mpl.colors.LinearSegmentedColormap.from_list('twilight_stacked', colors2)


offset = np.angle(qfft[numzones,0])+np.pi/2.0
if offset>np.pi:
    offset = offset-np.pi
qplot = q+8.0*(x[:,np.newaxis]+offset)

fig = plt.figure(figsize=(10.24,10.24), frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.imshow(np.fliplr(qplot), cmap=mymap2, vmin=-qbrange/2.0, vmax=qbrange/2.0)
ax.imshow(np.mod(np.fliplr(qplot) + 8*np.pi/3.0,2*8*np.pi/3.0)-(8*np.pi/3.0), cmap='twilight', vmin=-8*np.pi/3.0, vmax=8*np.pi/3.0)
ax.set_axis_off()
fig.add_axes(ax)
#plt.tight_layout(h_pad=0, w_pad=0)
fig.savefig('full.png', dpi=200)

# %% Compute SVD

u, s, vh = np.linalg.svd(q)

rank = 128
qr = u[:,:rank] @ (s[:rank,np.newaxis] * vh[:rank,:])


qrplot = qr+8.0*(x[:,np.newaxis]+offset)

fig = plt.figure(figsize=(10.24,10.24), frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
#ax.imshow(np.fliplr(qplot), cmap=mymap2, vmin=-qbrange/2.0, vmax=qbrange/2.0)
ax.imshow(np.mod(np.fliplr(qrplot) + 8*np.pi/3.0,2*8*np.pi/3.0)-(8*np.pi/3.0), cmap='twilight', vmin=-8*np.pi/3.0, vmax=8*np.pi/3.0)
ax.set_axis_off()
fig.add_axes(ax)
#plt.tight_layout(h_pad=0, w_pad=0)
fig.savefig('rank{}.png'.format(rank), dpi=200)