# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 18:17:51 2021

@author: maple
"""

import numpy as np
import scipy.linalg
import h5py
from numpy.polynomial import Polynomial
import scipy.optimize

import sys, os

sys.path.append(os.path.abspath('qg_dns/analysis/eigenvectors'))
from chm_utils import EigenvalueSolverFD

# %% Load data

case = 1
ampfile = np.load('dns_input/case{}/eigencomps_fd_smooth.npz'.format(case))
eigamps = ampfile['amps']
qbar = ampfile['qbar']

# %% Get eigenfunctions

nky = 8

eigsolver = EigenvalueSolverFD(qbar)
eigs = [None]*nky
for ky in range(1,nky+1):
    print(ky)
    eigs[ky-1] = eigsolver.solveEigenfunctions(ky=ky, norm='action')

# %% Pick out proper eigenfunctions

# This is the info that we need filled out
numeigs = 9
psiv = np.zeros((numeigs, 2048))
amps = np.zeros(numeigs)
#fits = [None]*numeigs
expfreqs = np.zeros(numeigs)
expphases = np.zeros(numeigs)

freqmult = np.zeros(numeigs, dtype=int)
eignums = np.zeros(numeigs, dtype=int)
kys = np.zeros(numeigs, dtype=int)

# This data is for plotting later
numsnaps = eigamps.shape[1]
mode0_phasedeviation = np.zeros(numsnaps)
eigenergies = np.zeros((numeigs, numsnaps))

# This is data for time-dependent deviations
phasedevs = np.zeros((numeigs, numsnaps))
ampdevs = np.ones((numeigs, numsnaps))

dt = 0.25
t = np.linspace(0, 64, num=numsnaps, endpoint=True)

# %% Compute the N most energetic eigenmodes

eigenergymult = np.zeros((nky, 2048))
for ky in range(1,len(eigs)+1):
    eigenergymult[ky-1,:] = -np.sum(eigs[ky-1]['vpsi']*eigs[ky-1]['vr'], axis=0)
    
actions = np.average(np.abs(eigamps[:,:,:])**2, axis=1)

energies = np.average(np.abs(eigamps[:,:,:])**2, axis=1) * eigenergymult
energyinds = np.argsort(-energies, axis=None)

for i in range(numeigs):
    eig = energyinds[i] % 2048
    ky = energyinds[i] // 2048 + 1
    
    kys[i] = ky
    eignums[i] = eig

    # Need to shift the phase of the amplitudes, since the FFT is in the domain [0,2pi]
    # while the real space coordinates are in the domain [-pi,pi]
    amp = eigamps[ky-1,:,eig] * (-1)**ky
    # Normalization factor for irfft
    amps[i] = np.sqrt(np.average(np.abs(amp)**2)) / 1024
    psiv[i,:] = np.real(eigs[ky-1]['vpsi'][:,eig])
    fit = Polynomial.fit(t,np.unwrap(np.angle(amp)), deg=1).convert()
    expfreqs[i] = fit.coef[1]
    expphases[i] = fit.coef[0]
    
    ampdevs[i,:] = (np.abs(amp)/1024) / amps[i]
    phasedevs[i,:] = np.unwrap(np.angle(amp)) - expfreqs[i]*t - expphases[i]
    
    eigenergies[i,:] = np.abs(amp)**2 * -np.sum(eigs[ky-1]['vpsi'][:,eig]*eigs[ky-1]['vr'][:,eig])
    if i == 0:
        mode0_phasedeviation = np.unwrap(np.angle(amp)) - expfreqs[0]*t - expphases[0]


def l1_dev(basefreq):
    totaldev = 0.0
    
    for i in range(numeigs):
        expfreq = expfreqs[i]
        amp = amps[i]
        
        fracparta = (expfreq / basefreq - np.round(expfreq / basefreq)) * basefreq
        
        totaldev = totaldev + np.abs(fracparta) * (amp / ky)
        
    return totaldev



freqRes = scipy.optimize.minimize_scalar(l1_dev, bounds=(-0.90/2.0, -0.78/2.0), method='bounded')
basefreq = freqRes.x
freqmult = np.array(list(map(int,np.round(expfreqs / basefreq))), dtype=int)
phases = expphases

# %% Output the input to the poincare section

savedata = { 'psiv': psiv, 'kys': kys, 'freqmult': freqmult, 'phases': phases, 'amps': amps, 'uy': eigsolver.uy, 'freq': freqRes.x, 'qbar': qbar }
np.savez('poincare_input/case{}_poincare_config_fd_smooth.npz'.format(case), **savedata)

# %% Time-dependent data for the poincare section

np.savez('poincare_input/case{}_eigencomponent_timedata.npz'.format(case), ampdevs=ampdevs, phasedevs=phasedevs)


# %% Save some extra data for plotting

avgenergy = np.sum(np.average(eigenergies, axis=1))
timeenergies = np.sum(eigenergies, axis=0)

np.savez('plot_scripts/case{}_eigencomponent_extradata.npz'.format(case), mode0_phasedeviation=mode0_phasedeviation, energydeviation=timeenergies/avgenergy)

# %% Save data for the validation test

"""
x = np.linspace(-np.pi, np.pi, num=2048)
qbar = np.cos(5*x)

eigsolver = EigenvalueSolverFD(qbar)

kys = np.array([3, 4], dtype=np.int32)
freqmult = np.array([3, 4], dtype=np.int32)

# This is the info that we need filled out
numeigs = len(kys)
psiv = np.zeros((numeigs, 2048))
amps = np.zeros(numeigs)
#fits = [None]*numeigs
expphases = np.zeros(numeigs)

t = np.linspace(0, 64, num=257, endpoint=True)

for i in range(numeigs):
    ky = kys[i]
    # Normalization factor for irfft
    amps[i] = 2 * np.pi
    psiv[i,:] = -np.cos(np.round(np.sqrt(25-ky**2))*x) / 2 / np.pi / 25
    expphases[i] = 0.0
    
phases = expphases


# %%

savedata = { 'psiv': psiv, 'kys': kys, 'freqmult': freqmult, 'phases': phases, 'amps': amps, 'uy': eigsolver.uy, 'freq': -8.0/25.0, 'qbar': qbar }
np.savez('poincare_input/poincare_config_validation.npz', **savedata)
"""