# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:36:07 2020

@author: maple
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.linalg
import h5py

# %%

simdata = h5py.File(r'C:\Users\maple\OneDrive\Research\CIMS\git\qg-edgeofchaos\dns_input\case2\snapshots_s1.h5', 'r')

# %% Load data

index = 0
q = simdata['tasks/q'][index,:,:]

nx = 2048
ny = 2048

kx = np.fft.fftfreq(nx, 1.0/nx)
ky = np.fft.rfftfreq(ny, 1.0/ny)
kxg, kyg = np.meshgrid(kx, ky, indexing='ij')

k2 = kxg**2 + kyg**2
invlap = np.zeros(k2.shape)
invlap[k2>0] = -1.0 / k2[k2>0]

padding = np.zeros((nx//2, nx//2+1))
rpadding = np.zeros((3*nx//2, nx//4))

# %% Functions used for computing the Poisson bracket

def upsample(f_fft):
    f1 = np.concatenate([1.5*f_fft[:nx//2,:], padding, 1.5*f_fft[nx//2:,:]], axis=0)
    return np.fft.irfft2(np.concatenate([1.5*f1, rpadding], axis=1))

def poisson_bracket(psifft, qfft):
    qx = upsample(1j*kxg*qfft)
    qy = upsample(1j*kyg*qfft)
    vx = upsample(-1j*kyg*psifft)
    vy = upsample(1j*kxg*psifft)

    pbfine = vx*qx + vy*qy
    pbfft = np.fft.rfft2(pbfine)
    pbfft1 = np.concatenate([pbfft[:nx//2,:nx//2+1]/(1.5*1.5), pbfft[-nx//2:,:nx//2+1]/(1.5*1.5)], axis=0)

    return pbfft1

# %% Resample then perform the Poisson bracket multiplication

qfft = np.fft.rfft2(q)
psifft = qfft * invlap

pbpsiqfft = poisson_bracket(psifft, qfft)

pbqspec = np.real(pbpsiqfft * np.conj(qfft)) * (1+(ky>0))  /257/2048/2048/2048/2048
qspec = np.abs(qfft)**2 * (1+(ky>0))  /257/2048/2048/2048/2048

# %%

kinds = np.argsort(np.ravel(k2))
k2_1d = np.arange(1,1024)**2
dqspec_1d = np.zeros(len(k2_1d))
pbqspec_1d = np.zeros(len(k2_1d))

qspec_sorted = np.ravel(qspec)[kinds]
pbqspec_sorted = np.ravel(pbqspec)[kinds]

k2_1dinds = np.searchsorted(np.ravel(k2)[kinds], k2_1d, side='right')

dqspec_1d[0] = np.sum(qspec_sorted[:k2_1dinds[0]])
pbqspec_1d[0] = np.sum(pbqspec_sorted[:k2_1dinds[0]])
for j in range(1, len(k2_1d)):
    dqspec_1d[j] = np.sum(qspec_sorted[k2_1dinds[j-1]:k2_1dinds[j]])
    pbqspec_1d[j] = np.sum(pbqspec_sorted[:k2_1dinds[j]])
#despec_1d = 
k_1d = np.sqrt(k2_1d)-0.5

# %%

plt.figure()

plt.semilogx(k_1d, pbqspec_1d*257)
plt.axvspan(14, 15)
plt.axhline(0)
plt.show()
