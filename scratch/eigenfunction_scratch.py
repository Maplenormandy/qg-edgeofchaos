# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:42:20 2022

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

import sys
import os

sys.path.append(os.path.abspath('../qg_dns/analysis/eigenvectors'))
from chm_utils import EigenvalueSolverFD, EigenvalueSolverFDNonSym, calcPixelEquivLatitude

# %% Load data

nx = 2048
x = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)

simdata = h5py.File('../dns_input/case1/snapshots_s2.h5', 'r')
q = simdata['tasks/q'][0,:,:]

# %% Plot zonal q

plt.figure()
for j in range(1):
    plt.plot(np.gradient(np.average(simdata['tasks/q'][j,:,:], axis=1) + 8*x))

# %% Compute qbar

qbar = calcPixelEquivLatitude(q, x)

smoother = np.exp(-96*(x+np.pi)) + np.exp(-96*(np.pi-x))
smoother = smoother / np.sum(smoother)

qbar_smooth = np.fft.irfft(np.fft.rfft(qbar)*np.fft.rfft(smoother))

qbar_smooth = np.average(simdata['tasks/q'][0,:,:], axis=1)

# %% Compute eigenfunctions

# eigsolver = EigenvalueSolverFD(qbar_smooth)
eigsolver = EigenvalueSolverFDNonSym(qbar_smooth)

nky = 4
rangeky = range(1,1+nky)
eigs = [None]*nky

print("Solving for eigenfunctions")
for ky in rangeky:
    print(ky)
    #eigs[ky-1] = eigsolver.solveEigenfunctions(ky=ky, norm='action')
    eigs[ky-1] = eigsolver.solveEigenfunctions(ky=ky)
    
# %% Plot eigenvalues

ky = 3
plt.scatter(np.real(eigs[ky-1]['w']), np.imag(eigs[ky-1]['w']))
    
# %% Radial plot

pod_re = np.load('../dns_input/case1/raw_podmode005.npy')
pod_im = np.load('../dns_input/case1/raw_podmode006.npy')

plt.imshow(pod_im)

pod_timetraces = np.load('../dns_input/case1/pod_timetraces.npz')['arr_0']

# %% 

#pod_fft = np.fft.fft(pod_re-1j*pod_im, axis=1)
#pod_radial_raw = pod_fft[:,-1]
#pod_radial = np.real(pod_radial_raw / np.fft.fft(pod_radial_raw)[1])

rank = 1
ind = np.argpartition(np.imag(eigs[ky-1]['w']), -rank)[-rank]

#ind = 1
eigplot = eigs[ky-1]['vr'][:,ind]

fig = plt.figure()
ax = fig.add_subplot(111)
ax2 = ax.twinx()
ax2.plot(eigsolver.uy, c='tab:blue')
ax.plot(np.real(eigplot), c='tab:orange')
ax.plot(np.imag(eigplot), c='tab:orange', ls='--')
#ax.plot(np.real(eigplot2), c='tab:orange')
#ax.plot(np.imag(eigplot2), c='tab:orange', ls='--')


#plt.plot(-pod_radial / np.sqrt(np.sum(pod_radial**2)))


# %%

t = np.linspace(0, 64, num=256, endpoint=False)
#plt.plot(pod_timetraces[:,1])
ph0 = np.unwrap(np.angle(pod_timetraces[:,3] + 1j * pod_timetraces[:,4]))
plt.plot(ph0)

fit = np.polynomial.polynomial.Polynomial.fit(np.linspace(0, 64, num=256, endpoint=False), ph0, deg=1).convert()
om0 = fit.coef[1]
