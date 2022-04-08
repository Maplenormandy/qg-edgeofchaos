# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 16:42:20 2022

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

import sys
import os

sys.path.append(os.path.abspath('../qg_dns/analysis/eigenvectors'))
from chm_utils import EigenvalueSolver, EigenvalueSolverFD, EigenvalueSolverFDNonSym, calcPixelEquivLatitude

# %% Load data

case = 2

nx = 2048
x = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)

simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, 3-case), 'r')

qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))


# %% Plot zonal q

plt.figure()
for j in range(1):
    plt.plot(np.gradient(np.average(simdata['tasks/q'][j,:,:], axis=1) + 8*x))

qbar = qbars['qbar'][0,:]

# %% Compute eigenfunctions

ky = 3
nt = simdata['tasks/q'].shape[0]
eigs = [None]*nt

print("Solving for eigenfunctions")
for i in range(nt):
    print(i)
    eigsolver = EigenvalueSolver(qbars['qbar'][i,:])
    eigs[i] = eigsolver.solveEigenfunctions(ky=ky, damping=True)

    
# %% Plot eigenvalues
cmap = mpl.cm.get_cmap('viridis')

fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax)

ax.plot(eigsolver.uy, range(nx))
ax.scatter(eigsolver.uy[eigsolver.b>8], np.arange(nx)[eigsolver.b>8], marker='x', c='tab:orange')

ax2.axhline(0, c='k', ls='--')
for i in range(nt):
    ax2.scatter(np.real(eigs[i]['w']), np.imag(eigs[i]['w']), color=cmap(i/nt))
    
ax2.set_ylim([-2*np.pi, 0.2*np.pi])

# %% Get eigenvalue amps
eigamps = np.zeros((2048, nt), dtype=complex)
for i in range(nt):
    q = simdata['tasks/q'][i,:,:]
    qffty = np.fft.rfft(q, axis=1)
    eigamps[:,i] = np.conj(eigs[i]['vl']).T @ qffty[:,ky]

# %% Sort eigenvalues and get amplitudes

eigsorts = [None]*nt

for i in range(nt):
    #eigsorts[i] = np.argsort(-np.abs(eigamps[:,i]))
    eigsorts[i] = np.argsort(np.real(eigs[i]['w']))
    #eigsorts[i] = np.argsort(-np.imag(eigs[i]['w']))

# %% How well are the eigenvectors preserved?

toplot = np.abs(np.conj((eigs[0]['vl'][:,eigsorts[0]]).T) @ eigs[1]['vr'][:,eigsorts[1]])

maxrow = np.argmax(toplot, axis=0)


plt.figure()
plt.imshow(np.clip(toplot, 0, 1), origin='lower')
plt.colorbar()

# %%

plt.figure()
plt.imshow(np.clip(np.abs(eigs[0]['vpsi'][:,eigsorts[0]]), 0, 0.01))

  
# %% Radial plot

ind = 0
eig = eigsorts[ind][38]

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(np.real(eigs[ind]['vpsi'][:,eig]), c='tab:blue')
ax.plot(np.imag(eigs[ind]['vpsi'][:,eig]), c='tab:blue', ls='--')

axt = ax.twinx()
axt.plot(eigsolver.uy, c='tab:orange')

ax2 = fig.add_subplot(212)
ax2.plot(np.real(eigamps[eig,:]), c='tab:blue')
ax2.plot(np.imag(eigamps[eig,:]), c='tab:blue', ls='--')