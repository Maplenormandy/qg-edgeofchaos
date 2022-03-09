# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 18:17:51 2021

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import h5py
import matplotlib as mpl
from numpy.polynomial import Polynomial
import scipy.optimize
import scipy.signal

import sys, os
sys.path.append(os.path.abspath('../qg_dns/analysis/eigenvectors'))
from chm_utils import EigenvalueSolverFD

# %% Load data

case = 2
ampfile = np.load('../dns_input/case{}/eigencomps_fd_smooth.npz'.format(case))
eigamps = ampfile['amps']
qbar = ampfile['qbar']

dt = 0.25


# %%

nky = 8
eigsolver = EigenvalueSolverFD(qbar)
eigs = [None]*nky
for ky in range(1,nky+1):
    print(ky)
    try:
        eigs[ky-1] = np.load('case{}_eigsolver_ky{}.npz'.format(case, ky))
        print("Loaded")
    except:
        print("Solving")
        eigs[ky-1] = eigsolver.solveEigenfunctions(ky=ky, norm='action')
        np.savez('case{}_eigsolver_ky{}.npz'.format(case, ky), eigs[ky-1])
    
# %% Compute DMD
    
ky = 1
eignum0 = 4
eignum1 = 4

states = 4

"""
xmat = np.zeros((states*2, eigamps.shape[1]-states), dtype=complex)
ymat = np.zeros((states*2, eigamps.shape[1]-states), dtype=complex)

for i in range(states):
    if i == 0:
        ymat[-1 - i*2] = eigamps[ky-1,states-i:,eignum0]
        ymat[-2 - i*2] = eigamps[ky-1,states-i:,eignum1]
    else:
        ymat[-1 - i*2] = eigamps[ky-1,states-i:-i,eignum0]
        ymat[-2 - i*2] = eigamps[ky-1,states-i:-i,eignum1]
    xmat[-1 - i*2] = eigamps[ky-1,states-i-1:-(i+1),eignum0]
    xmat[-2 - i*2] = eigamps[ky-1,states-i-1:-(i+1),eignum1]
"""


xmat = np.zeros((states, eigamps.shape[1]-states), dtype=complex)
ymat = np.zeros((states, eigamps.shape[1]-states), dtype=complex)

for i in range(states):
    if i == 0:
        ymat[-1 - i] = eigamps[ky-1,states-i:,eignum0]
    else:
        ymat[-1 - i] = eigamps[ky-1,states-i:-i,eignum0]
    xmat[-1 - i] = eigamps[ky-1,states-i-1:-(1+i),eignum0]


u0, s0, vh0 = np.linalg.svd(xmat, full_matrices=False)

rank = 7
polar = False

u = u0
s = s0
vh = vh0

amat0 = np.conj(u.T) @ ymat @ np.conj(vh.T) @ np.diag(1.0 / s)
amat = amat0
w, vl, vr = scipy.linalg.eig(amat, left=True)

dmdmodes = ymat @ np.conj(vh.T) @ np.diag(1.0 / s) @ vr @ np.diag(1.0 / w)
dmdleftmodes = u @ vl
dmdtraces = np.conj(dmdleftmodes.T) @ xmat

plt.figure()
plt.plot(dmdtraces.T)
print(np.imag(np.log(w))/0.25)

