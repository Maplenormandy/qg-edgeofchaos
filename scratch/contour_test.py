# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:53:22 2022

@author: maple
"""

import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import scipy.interpolate

# %% Load data

case = 1
data = np.load('../poincare_input/case{}_poincare_config_fd_smooth.npz'.format(case))

qbar = data['qbar']
        
uy = data['uy']
psiv = data['psiv']
freq = data['freq']
freqs = freq*data['freqmult']
phases = data['phases']
kys = data['kys']
amps = data['amps']
numeigs = len(kys)

amps_mod = amps*1.2
phases_mod = phases
zonalmult = 1.0

nx = psiv.shape[1]
kx = np.fft.rfftfreq(nx, 1.0/nx)
x = np.linspace(-np.pi,np.pi, num=nx, endpoint=False)

pad = 4
xp = np.zeros(nx+2*pad)
xp[pad:-pad] = x
xp[:pad] = x[-pad:] - 2*np.pi
xp[-pad:] = x[:pad] + 2*np.pi

utildey = np.fft.irfft(-1j*kx[np.newaxis,:]*np.fft.rfft(psiv, axis=1), axis=1)
qtilde = np.fft.irfft(-(kx[np.newaxis,:]**2 + kys[:,np.newaxis]**2)*np.fft.rfft(psiv, axis=1), axis=1)

def circularInterpolant(vec):
    vecp = np.zeros(nx+2*pad)
    vecp[pad:-pad] = vec
    vecp[:pad] = vec[-pad:]
    vecp[-pad:] = vec[:pad]
    
    return scipy.interpolate.interp1d(xp, vecp, kind='quadratic')

psif = [circularInterpolant(psiv[i,:]) for i in range(numeigs)]
utyf = [circularInterpolant(utildey[i,:]) for i in range(numeigs)]
qtf = [circularInterpolant(qtilde[i,:]) for i in range(numeigs)]
qbarf = circularInterpolant(qbar + 8*x)

# Compute desried contour
qts = np.array(list((qtf[i](x)[:,np.newaxis])*(np.cos(kys[i]*x - phases_mod[i])[np.newaxis,:])*amps_mod[i] for i in range(numeigs)))
qplot = (qbar*zonalmult + 8*x)[:,np.newaxis] + np.sum(qts, axis=0)

# %%

plt.imshow(qplot, origin='lower')

# %%

nparticles = 97
xsamples = np.linspace(-np.pi, np.pi, num=nparticles, endpoint=False)
qsamples = qbarf(xsamples)

z0 = np.ones(nparticles*2) * 8000.0

for i in range(nparticles):
    qcont = qsamples[i]
    
    if xsamples[i] < -3*np.pi/4:
        contours1 = measure.find_contours(qplot, qcont)
        contours2 = measure.find_contours(qplot, qcont+2*8*np.pi)
        contours = contours1 + contours2
    elif xsamples[i] > 3*np.pi/4:
        contours1 = measure.find_contours(qplot, qcont)
        contours2 = measure.find_contours(qplot, qcont-2*8*np.pi)
        contours = contours1 + contours2
    else:
        contours = measure.find_contours(qplot, qcont)
        
    print("{} - {} contours".format(qcont, len(contours)))
    for c in contours:
        if np.abs(c[0,1] - c[-1,1]) < 1.0:
            continue
        
        c2 = (2*np.pi*c/2048) - np.pi
        
        ymin = np.argmin(c2[:,1])
        if c2[ymin,1] < z0[nparticles+i]:
            z0[i] = c2[ymin,0]
            z0[nparticles+i] = c2[ymin,1]
            
# %%

plt.figure()
plt.scatter(z0[nparticles:], z0[:nparticles])