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

nx = 2048
xplot = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)

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

eigrange = np.arange(30, 80, dtype=int)


ky = 5
#trunceigs = 7

delay = 1
ymat = (eigamps[ky-1,delay:,:].T)[eigrange,:]
xmat = (eigamps[ky-1,:-delay,:].T)[eigrange,:]

u0, s0, vh0 = np.linalg.svd(xmat, full_matrices=False)

rank = 4
polar = False

u = u0[:,:rank]
s = s0[:rank]
vh = vh0[:rank,:]

amat0 = np.conj(u.T) @ ymat @ np.conj(vh.T) @ np.diag(1.0 / s)
if polar:
    amat, p = scipy.linalg.polar(amat0)
else:
    amat = amat0
w, vl, vr = scipy.linalg.eig(amat, left=True)

dmdmodes = ymat @ np.conj(vh.T) @ np.diag(1.0 / s) @ vr @ np.diag(1.0 / w)
dmdleftmodes = u @ vl
dmdtraces = np.conj(dmdleftmodes.T) @ xmat

# Compute rsquared of traces
xt = dmdtraces[:,:-delay]
yt = dmdtraces[:,delay:]
at = np.sum(yt * np.conj(xt), axis=1) / (np.sum(xt*np.conj(xt), axis=1))
rst = 1.0 - np.sum(np.abs(yt - xt*at[:,np.newaxis])**2, axis=1) / np.sum(np.abs(yt)**2, axis=1)


energy = np.sum(np.abs(vr * s[:,np.newaxis])**2, axis=0)
ind = np.argsort(-rst)

rsquared = 1.0 - np.sum(np.abs(ymat - (u @ amat @ np.diag(s) @ vh))**2) / np.sum(np.abs(ymat**2))
print(rsquared)

fig = plt.figure()

maxplot = 4
plotrank = min((maxplot,rank))
for i in range(plotrank):
    ax = plt.subplot(maxplot, 2, i*2+1)

    dmdnum = ind[i]
    
    eigcorresp = np.argmax(np.abs(dmdmodes[:,dmdnum]))
    dmdamp = dmdmodes[eigcorresp,dmdnum]
    dmdplot = (eigs[ky-1]['vpsi'][:,eigrange] @ dmdmodes[:,dmdnum]) / (dmdamp / np.abs(dmdamp))
    
    ax.plot(np.real(dmdplot), c='tab:blue')
    ax.plot(np.imag(dmdplot), c='tab:blue', ls='--')
    ax.plot(eigs[ky-1]['vpsi'][:,eigrange[eigcorresp]]*np.abs(dmdamp), c='tab:orange')
    
    ax2 = plt.subplot(maxplot, 2, i*2+2)
    ax2.plot(np.real(dmdtraces[dmdnum,:]), c='tab:blue')
    ax2.plot(np.imag(dmdtraces[dmdnum,:]), c='tab:blue', ls='--')
    ax2.set_title('r2 = {}'.format(rst[dmdnum]))