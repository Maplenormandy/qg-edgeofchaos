# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:47:18 2021

@author: maple
"""

import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

# %%

case = 1
simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, 3-case), 'r')

# %% Load data

index = 0
q = simdata['tasks/q'][index,:,:]


qbar_zonal = np.average(q,axis=1)

nx = 2048
x = np.linspace(-np.pi,np.pi, num=nx, endpoint=False)

qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))

qbar_equiv = qbars['qequiv'][0,:] - 8*x
qbar = qbars['qbar'][0,:]

qb = q+8*x[:,np.newaxis]
#qb = np.zeros(q.shape)+8*x[:,np.newaxis]

# %% Add an extra pixel to the edges

qp = np.zeros([qb.shape[0]+1, qb.shape[1]+1])
qp[:nx,:nx] = qb
qp[nx,:nx] = qb[0,:]+2*np.pi*8
qp[:nx,nx] = qb[:,0]
qp[nx,nx] = qb[0,0]+2*np.pi*8

#plt.imshow(qp)
maxqp = np.max(qp)
minqp = np.min(qp)
minsafe = maxqp-2*8*np.pi
maxsafe = minqp+2*8*np.pi

# %% 

levels = np.linspace(-8*np.pi, 8*np.pi, num=521, endpoint=False)
levels = levels+np.mean(np.diff(levels))/2
areas = np.zeros(len(levels))
numContours = np.zeros(len(levels))

for ind in range(len(levels)):
    
    # Compute contours
    if levels[ind] <= minsafe:
        contours1 = measure.find_contours(qp, levels[ind])
        contours2 = measure.find_contours(qp, levels[ind]+2*8*np.pi)
        
        for i in range(len(contours2)):
            contours2[i][:,0] -= nx
        
        rawcontours = contours1+contours2
        
    elif levels[ind] >= maxsafe:
        contours1 = measure.find_contours(qp, levels[ind])
        contours2 = measure.find_contours(qp, levels[ind]-2*8*np.pi)
        
        for i in range(len(contours2)):
            contours2[i][:,0] += nx
        
        rawcontours = contours1+contours2
    else:
        rawcontours = measure.find_contours(qp, levels[ind])
    
    rawcontours.sort(key=lambda c: c[0,1])
    
    contours = []
    
    next_ind = np.ones(len(rawcontours), dtype=int)*-1
    added = np.zeros(len(rawcontours), dtype=bool)
    
    
    # Start stiching contours together
    for i in range(len(rawcontours)):
        c = rawcontours[i]
        
        for j in range(len(rawcontours)):
            c2 = rawcontours[j]
            
            if np.all(np.abs(np.mod(c[-1,:]-c2[0,:] + nx/2.0, nx)-nx/2.0)<1e-5):
                next_ind[i] = j
                
    for i in range(len(rawcontours)):
        c = rawcontours[i]
        
        if added[i]:
            continue
        
        if next_ind[i] >= 0:
            j = next_ind[i]
            
            while j != i and j >= 0:
                c2 = rawcontours[j]
                c = np.append(c, c2[1:,:], axis=0)
                added[j] = True
                j = next_ind[j]
                
            if j == i:
                c = c[:-1,:]
        
        contours.append(np.unwrap(c, axis=0, discont=nx/2.0))
        added[i] = True
        
    num_encircling = 0
        
    for i in range(len(contours)):
        c = contours[i]
        cx = c[:,0]
        cx[:-1] += c[1:,0]
        cx[-1] += c[0,0]
        
        cx = (cx * (2*np.pi/2.0/nx) - np.pi)
        
        dy = (np.mod(np.diff(c[:,1], append=c[0,1])+nx/2.0, nx)-nx/2.0) / nx * 2*np.pi
        
        areas[ind] += np.sum(cx*dy)
        
        if np.abs(c[0,1] - c[-1,1]) > nx/2.0:
            num_encircling += 1
        
    print(ind, num_encircling)
# %%

xareas = areas / 2 / np.pi

levelsbar = levels - 8*xareas

uyareas = np.cumsum(levelsbar*np.gradient(xareas) - np.average(levelsbar*np.gradient(xareas)))
uy = np.cumsum(qbar_equiv/nx*2*np.pi - np.average(qbar_equiv/nx*2*np.pi))

b = np.gradient(qbar_equiv) / (2*np.pi/nx)
bareas = np.gradient(levelsbar) / np.gradient(xareas)
bz = np.gradient(np.average(qbars['qbar'], axis=0)) / (2*np.pi/nx)

#plt.plot(xareas, -(uyareas - np.average(uyareas, weights=np.gradient(xareas))), marker='.')
#plt.plot(x, -(uy - np.average(uy)))

plt.plot(xareas, bareas, marker='.')
plt.plot(x, b)
plt.plot(x, bz)
plt.axhline(-8)