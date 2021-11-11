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

from eigensolvers import EigenvalueSolverFD

# %% Load data

ampfile = np.load('dns_input/amps_fd_smooth.npz')
eigamps = ampfile['amps']
qbar = ampfile['qbar']

longtimefile = np.load('dns_input/longtime_amps_fd_smooth.npz')
eigampslt = longtimefile['amps']

# %% Get eigenfunctions

nky = 3

eigsolver = EigenvalueSolverFD(qbar)
eigs = [None]*nky
for ky in range(1,nky+1):
    print(ky)
    eigs[ky-1] = eigsolver.solveEigenfunctions(ky=ky)

# %% Pick out proper eigenfunctions

# Here we have the 9 most energetic eigenmodes
eignums = np.array([5, 3, 1, 4, 2, 3, 4, 6, 0], dtype=np.int32)
kys =     np.array([1, 1, 3, 2, 3, 2, 1, 1, 2], dtype=np.int32)
freqmult= np.array([2, 4, 6, 5, 6, 5, 4, 2, 10], dtype=np.int32)

"""
# Here we pair up eigenfunctions by their energetic pair
eignums = np.array([5,6, 3,4, 1,2, 4,3, 0, 3,4, 5,6, 1,2, 0], dtype=np.int32)
kys =     np.array([1,1, 1,1, 3,3, 2,2, 2, 3,3, 2,2, 2,2, 3], dtype=np.int32)
freqmult= np.zeros(len(eignums), dtype=np.int32)

# Here we only keep eigenfunctions where amplitude never drops below std, and their sin/cos partner
eignums = np.array([5,6, 3,4, 1,2, 0], dtype=np.int32)
kys =     np.array([1,1, 1,1, 3,3, 2], dtype=np.int32)
freqmult= np.array([1,1, 2,2, 3,3, 5], dtype=np.int32)
"""

# This is the info that we need filled out
numeigs = len(eignums)
psiv = np.zeros((numeigs, 2048))
amps = np.zeros(numeigs)
#fits = [None]*numeigs
expfreqs = np.zeros(numeigs)
expphases = np.zeros(numeigs)

# This data is for plotting later
mode0_phasedeviation = np.zeros(257)
eigenergies = np.zeros((numeigs, 257))

# This is data for time-dependent deviations
phasedevs = np.zeros((numeigs, 257))
ampdevs = np.ones((numeigs, 257))
phasedevslt = np.zeros((numeigs, eigampslt.shape[1]))
ampdevslt = np.ones((numeigs, eigampslt.shape[1]))

dt = 0.25


t = np.linspace(0, 64, num=257, endpoint=True)

for i in range(numeigs):
    ky = kys[i]
    eig = eignums[i]
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
    
    amplt = eigampslt[ky-1,:,eig]
    ampdevslt[i,:] = (np.abs(amplt)/1024) / amps[i]
    phasedevslt[i,:] = np.unwrap(np.angle(amplt)) - expphases[i]
    
    eigenergies[i,:] = np.abs(amp)**2 * -np.sum(eigs[ky-1]['vpsi'][:,eig]*eigs[ky-1]['vr'][:,eig])
    if i == 0:
        mode0_phasedeviation = np.unwrap(np.angle(amp)) - expfreqs[0]*t - expphases[0]


def l1_dev(basefreq):
    totaldev = 0.0
    
    for i in range(numeigs):
        expfreq = expfreqs[i]
        amp = amps[i]
        
        fracparta = (expfreq / basefreq - np.round(expfreq / basefreq)) * basefreq
        
        totaldev = totaldev + np.abs(fracparta) * amp
        
    return totaldev


freqDelta = lambda x: np.sum(np.abs(freqmult*x - expfreqs)*amps)
freqRes = scipy.optimize.minimize_scalar(freqDelta, bounds=(1.2*expfreqs[0]/freqmult[0], 0.8*expfreqs[0]/freqmult[0]))
phases = expphases

# %% Output the input to the poincare section

savedata = { 'psiv': psiv, 'kys': kys, 'freqmult': freqmult, 'phases': phases, 'amps': amps, 'uy': eigsolver.uy, 'freq': freqRes.x, 'qbar': qbar }
np.savez('poincare_input/poincare_config_fd_smooth.npz', **savedata)

# %% Time-dependent data for the poincare section

np.savez('poincare_input/eigencomponent_timedata.npz', ampdevs=ampdevs, phasedevs=phasedevs)
np.savez('poincare_input/eigencomponent_longtimedata.npz', ampdevs=ampdevslt, phasedevs=phasedevslt)


# %% Save some extra data for plotting

avgenergy = np.sum(np.average(eigenergies, axis=1))
timeenergies = np.sum(eigenergies, axis=0)

np.savez('plot_scripts/eigencomponent_extradata.npz', mode0_phasedeviation=mode0_phasedeviation, energydeviation=timeenergies/avgenergy)

# %% Save data for the validation test

x = np.linspace(-np.pi, np.pi, num=2048)
qbar = np.cos(5*x)

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
