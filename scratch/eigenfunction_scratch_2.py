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

#simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, 3-case), 'r')

qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))



# %% Compute eigenfunctions

nky = 4
eigs = [None]*nky

print("Solving for eigenfunctions")
for ky in range(1,nky+1):
    print(ky)
    eigsolver = EigenvalueSolverFD(np.average(qbars['qbar'], axis=0))
    eigs[ky-1] = eigsolver.solveEigenfunctions(ky=ky)

    
# %% Plot eigenvalues
cmap = mpl.cm.get_cmap('viridis')

fig = plt.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212, sharex=ax)

ax.plot(eigsolver.uy, range(nx))
ax.scatter(eigsolver.uy[eigsolver.b>8], np.arange(nx)[eigsolver.b>8], marker='x', c='tab:orange')

ax2.axhline(0, c='k', ls='--')
for i in range(nky):
    ax2.scatter(np.real(eigs[i]['w']), np.ones(eigs[i]['w'].shape)*(i+1))
    
#ax2.set_ylim([-2*np.pi, 0.2*np.pi])


  
# %% Radial plot

ind = 1
eig = 230

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.real(eigs[ind]['vpsi'][:,eig]), c='tab:blue')
ax.plot(np.imag(eigs[ind]['vpsi'][:,eig]), c='tab:blue', ls='--')

axt = ax.twinx()
axt.plot(eigsolver.uy, c='tab:orange')
