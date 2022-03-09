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

case = 2
ampfile = np.load('dns_input/case{}/eigencomps_fd_smooth.npz'.format(case))
eigamps = ampfile['amps']
qbar = ampfile['qbar']
suffix = '_uphavg'
usemin = False

# %% Get eigenfunctions

nky = 8

eigsolver = EigenvalueSolverFD(qbar)
eigs = [None]*nky
for ky in range(1,nky+1):
    print(ky)
    try:
        eigs[ky-1] = np.load('scratch/case{}_eigsolver_ky{}.npz'.format(case, ky))
        print("Loaded")
    except:
        print("Solving")
        eigs[ky-1] = eigsolver.solveEigenfunctions(ky=ky, norm='action')
        np.savez('scratch/case{}_eigsolver_ky{}.npz'.format(case, ky), **eigs[ky-1])

# %% Compute the coherence of the eigenfunctions

numofs = 1
rsqt = np.zeros((eigamps.shape[0], numofs, eigamps.shape[2]))

for i in range(numofs):
    fitofs = i+1
    
    x = eigamps[:,:-fitofs,:]
    y = eigamps[:,fitofs:,:]
    
    amat = np.sum(y * np.conj(x), axis=1) / np.sum(np.abs(x)**2, axis=1)
    
    residuals = y - (x * amat[:,np.newaxis,:])
    vartot = np.average(np.abs(y)**2, axis=1)
    varresid = np.average(np.abs(residuals)**2, axis=1)
    
    rsqt[:,i,:] = 1 - (varresid/vartot)

rsquaredall = np.min(rsqt, axis=1)
rsqinds = np.argsort(-rsquaredall, axis=None)

eigenergies = np.zeros((nky, 2048))
for ky in range(1,len(eigs)+1):
    eigenergies[ky-1,:] = -np.sum(eigs[ky-1]['vpsi']*eigs[ky-1]['vr'], axis=0)

energies = np.average(np.abs(eigamps[:,:,:])**2, axis=1) * eigenergies
minenergies = np.min(np.abs(eigamps[:,:,:])**2, axis=1) * eigenergies
energyinds = np.argsort(-energies, axis=None)
minenergyinds = np.argsort(-minenergies, axis=None)

kyeig = np.zeros((eigamps.shape[0], eigamps.shape[2]), dtype=int)
kyeig[:] = np.arange(1,nky+1, dtype=int)[:,np.newaxis]
# This is a hacky way to estimate kx
kxeig = np.zeros((eigamps.shape[0], eigamps.shape[2]), dtype=int)
kxeig[:] = np.arange(0,eigamps.shape[2], dtype=int)[np.newaxis,:]
kxeig = (kxeig + 1) // 2

k2eig = kyeig*kyeig + kxeig*kxeig
scaleind = np.argsort(k2eig, axis=None, kind='stable')

uphs = np.array([eigs[ky-1]['w'] for ky in range(1,len(eigs)+1)])
uphinds = np.argsort(uphs, axis=None)

inds = uphinds


# %% Pick out the eigenfunctions we're going to use

if case == 1:
    numeigs = 8
else:
    numeigs = 48

# This is the info that we need filled out

print("Number of eigenfunctions: ", numeigs)

psiv = np.zeros((numeigs, 2048))
amps = np.zeros(numeigs)
#fits = [None]*numeigs
expfreqs = np.zeros(numeigs)
expphases = np.zeros(numeigs)

rsquared = np.zeros(numeigs)

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

# Data for testing
freqmulttest = np.zeros(numeigs, dtype=int)
eigfreqs = np.zeros(numeigs)

dt = 0.25
t = np.linspace(0, 64, num=numsnaps, endpoint=True)



for i in range(numeigs):
    eig = inds[i] % 2048
    ky = inds[i] // 2048 + 1
    
    kys[i] = ky
    eignums[i] = eig
    rsquared[i] = rsquaredall[ky-1, eig]

    # Need to shift the phase of the amplitudes, since the FFT is in the domain [0,2pi]
    # while the real space coordinates are in the domain [-pi,pi]
    amp = eigamps[ky-1,:,eig] * (-1)**ky
    # Normalization factor for irfft
    if usemin:
        amps[i] = np.sqrt(np.min(np.abs(amp)**2)) / 1024
    else:
        amps[i] = np.sqrt(np.average(np.abs(amp)**2)) / 1024
    
    psiv[i,:] = np.real(eigs[ky-1]['vpsi'][:,eig])
    fit = Polynomial.fit(t,np.unwrap(np.angle(amp)), deg=1).convert()

    expfreqs[i] = fit.coef[1]
    expphases[i] = fit.coef[0]
    
    ampdevs[i,:] = (np.abs(amp)/1024) / amps[i]
    phasedevs[i,:] = np.unwrap(np.angle(amp)) - expphases[i]
    
    eigenergies[i,:] = np.abs(amp)**2 * -np.sum(eigs[ky-1]['vpsi'][:,eig]*eigs[ky-1]['vr'][:,eig])
    if i == 0:
        mode0_phasedeviation = np.unwrap(np.angle(amp)) - expfreqs[0]*t - expphases[0]

    freqmulttest[i] = int(np.round(ky / k2eig[ky-1,eig] * 20))
    eigfreqs[i] = eigs[ky-1]['w'][eig]*ky

# Set doppler shift
ind_dop = np.where(inds==energyinds[0])[0][0]

ky_dop = kys[ind_dop]
dopplerc = expfreqs[ind_dop] / ky_dop
#dopplerc = 0

expfreqs = expfreqs - kys*dopplerc


def l1_dev(basefreq):
    totaldev = 0.0
    
    for i in range(numeigs):
        expfreq = expfreqs[i]
        amp = amps[i]
        
        fracparta = (expfreq / basefreq - np.round(expfreq / basefreq)) * basefreq
        
        totaldev = totaldev + np.abs(fracparta) * (amp / ky)**2
        
    return totaldev

freqsearchfunc = lambda x: l1_dev(-np.exp(x))

freqRes = scipy.optimize.minimize_scalar(l1_dev, bounds=(-0.90/2.0, -0.78/2.0), method='bounded')
basefreq = freqRes.x

#freqRes = scipy.optimize.minimize_scalar(freqsearchfunc, bounds=(np.log(0.06), np.log(0.08)), method='bounded')
#basefreq = -np.exp(freqRes.x)

#dopplerc = 2*basefreq
#expfreqs = expfreqs - kys*dopplerc

freqmult = np.array(list(map(int,np.round(expfreqs / basefreq))), dtype=int)
#freqmult = freqmulttest
phases = expphases

uph_exp = (expfreqs + kys*dopplerc) / kys
uph_fit = (freqmult*basefreq + kys*dopplerc) / kys

#print('residuals:', uph_fit-uph_exp)

# %% Output the input to the poincare section

print('saving: ' + 'poincare_input/case{}_poincare_config_fd_smooth{}.npz'.format(case, suffix))

savedata = { 'psiv': psiv, 'kys': kys, 'freqmult': freqmult, 'phases': phases, 'amps': amps, 'uy': eigsolver.uy, 'freq': freqRes.x, 'qbar': qbar, 'rsquared': rsquared, 'eignums': eignums, 'dopplerc': dopplerc }
np.savez('poincare_input/case{}_poincare_config_fd_smooth{}.npz'.format(case, suffix), **savedata)


np.savez('poincare_input/case{}_eigencomponent_timedata{}.npz'.format(case, suffix), ampdevs=ampdevs, phasedevs=phasedevs)


avgenergy = np.sum(np.average(eigenergies, axis=1))
timeenergies = np.sum(eigenergies, axis=0)

np.savez('plot_scripts/case{}_eigencomponent_extradata{}.npz'.format(case, suffix), mode0_phasedeviation=mode0_phasedeviation, energydeviation=timeenergies/avgenergy)




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