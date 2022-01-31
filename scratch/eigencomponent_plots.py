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

import sys, os
sys.path.append(os.path.abspath('../qg_dns/analysis/eigenvectors'))
from chm_utils import EigenvalueSolverFD

# %% Load data

ampfile = np.load('../dns_input/case1/eigencomps_fd_smooth.npz')
eigamps = ampfile['amps']
qbar = ampfile['qbar']

dt = 0.25


# %%

nky = 8
eigsolver = EigenvalueSolverFD(qbar)
eigs = [None]*nky
for ky in range(1,nky+1):
    print(ky)
    eigs[ky-1] = eigsolver.solveEigenfunctions(ky=ky, norm='action')
    
# %% Compute the number of non-resonant modes
    
uymin = np.min(eigsolver.uy)
for ky in range(1,nky+1):
    print(ky, np.sum(eigs[ky-1]['w'] < uymin))

# %% Energy per mode
    
eigenergies = np.zeros((nky, 2048))
for ky in range(1,len(eigs)+1):
    eigenergies[ky-1,:] = -np.sum(eigs[ky-1]['vpsi']*eigs[ky-1]['vr'], axis=0)

ky=1
plt.figure()
plt.plot(eigs[ky-1]['w'], eigenergies[ky-1,:], marker='.', ls='')


# %% Compute DMD frequencies of individual modes

s2 = np.sum(np.abs(eigamps[:,:-1,:])**2, axis=1)
amat = np.sum(eigamps[:,1:,:]*np.conj(eigamps[:,:-1,:]), axis=1) / s2
dmdfreqs = np.log(amat) / dt

actions = np.average(np.abs(eigamps[:,:,:])**2, axis=1)
std_actions = np.std(np.abs(eigamps[:,:,:])**2, axis=1)

energies = np.average(np.abs(eigamps[:,:,:])**2, axis=1) * eigenergies
#energies = np.quantile(np.abs(eigamps[:,:,:])**2, 0.15, axis=1) * eigenergies

#inds = np.argsort(-dmdfreqs, axis=None)
inds = np.argsort(-energies, axis=None)

coherence = (np.average(np.abs(eigamps[:,:,:])**2, axis=1) / np.std(np.abs(eigamps[:,:,:])**2, axis=1))
coherent = np.all(np.abs(eigamps[:,:,:])**2 > np.std(np.abs(eigamps[:,:,:])**2, axis=1)[:,np.newaxis,:], axis=1)

#inds = np.argsort(-coherence, axis=None)

# %%

nrows = 8
ncols = 4
f, ax = plt.subplots(nrows, ncols) #, gridspec_kw={'width_ratios' : [3,1,3,1,3,1]})

t = np.linspace(0, 64, num=257, endpoint=True)
#kfft = np.fft.rfftfreq(2048, d=1.0/2048)

for i in range(nrows*ncols):
    j = i%nrows
    k = (i//nrows)
    
    eig = inds[i] % 2048
    ky = inds[i] // 2048 + 1
    kx = np.argmax(np.abs(np.fft.rfft(np.real(eigs[ky-1]['vl'][:,eig]))))
    ax[j,k].set_title('ky={}, kx={}, eig={}'.format(ky,kx,eig))
    #ax.set_aspect('equal')
    ax[j,k].plot(t, np.real(eigamps[ky-1,:,eig]), c='#1f77b4')
    ax[j,k].plot(t, np.imag(eigamps[ky-1,:,eig]), ls='--', c='#1f77b4')
    ax[j,k].plot(t, np.abs(eigamps[ky-1,:,eig]), c='#2ca02c')
    ax[j,k].plot(t, -np.abs(eigamps[ky-1,:,eig]), c='#2ca02c')
    
    
    
    #avgamp = np.sqrt(energies[ky-1, eig]/257.0)
    #initialphase = np.angle(eigamps[ky-1,0,eig])
    #eigplot = -avgamp*np.cos(np.real(eigs[ky-1]['w'][eig])*t*ky - initialphase)
    
    #ax[j,k].plot(t, eigplot, c='#ff7f0e')
    
    #ax[j,k].axhline(ls='--', c='k')
    ax[j,k].set_xticklabels([])
    
    #ax[j,k+1].plot(np.real(eigamps[ky-1,:,eig]), np.imag(eigamps[ky-1,:,eig]), lw=1)
    #ax[j,k+1].scatter([0],[0], c='k', marker='+')
    #ax[j,k+1].set_aspect('equal')
    #ax[j,k+1].set_xticks([])
    #ax[j,k+1].set_yticks([])
    #ax[j,k+1].axis('off')

# %%
    
f, ax = plt.subplots(nrows, ncols*2)

for i in range(nrows*ncols):
    j = i%nrows
    k = (i//nrows)*2
    
    eig = inds[i] % 2048
    ky = inds[i] // 2048 + 1
    
    
    amps = np.abs(eigamps[ky-1,:,eig])**2
    ax[j,k].hist(amps, bins=24)
    ax[j,k].axvline(np.average(amps), ls='--', c='C2')
    ax[j,k].axvline(np.std(amps), ls='--', c='C1')
    ax[j,k].axvline(np.exp(np.average(np.log(amps))), ls='--', c='C3')
    
    freqs = np.diff(np.unwrap(np.angle(eigamps[ky-1,:,eig])))/dt
    ax[j,k+1].hist(freqs, bins=24)
    ax[j,k+1].axvline(np.real(eigs[ky-1]['w'][eig])*ky, ls='--', c='C2')
    if np.any(freqs>0):
        ax[j,k+1].axvline(0, ls='--', c='k')
        
    # Try DMD method of extracting frequency
    xmat = eigamps[ky-1,:-1,eig]
    ymat = eigamps[ky-1,1:,eig]
    s = np.sqrt(np.sum(np.abs(xmat)**2))
    amat = np.dot(ymat, np.conj(xmat)) / s**2
    dmdfreq = np.log(amat)/dt
    #ax[j,k+1].axvline(np.real(dmdfreq), ls='--', c='C1')
    ax[j,k+1].axvline(np.imag(dmdfreq), ls='--', c='C3')
        
    

# %%

"""
plt.figure()
plt.plot(np.real(eigs[0]['vpsi'][:,5]))

# %%

#plt.figure()
psiffty = np.zeros((2048,1025), dtype=np.complex)
qffty = np.zeros((2048,1025), dtype=np.complex)
psiffty[:,1] = eigs[0]['vpsi'][:,5]
qffty[:,1] = eigs[0]['vr'][:,5]
psiplot = np.fft.irfft(psiffty, axis=1)
qplot = np.fft.irfft(qffty, axis=1)

print('Fourier Transform Factor: ' + str(np.max(np.abs(psiffty))/np.max(psiplot)))
print('L2 norm factor: ' + str(-np.average(psiplot * qplot)*2048*2048*2048/4))

# %% Cumulative energy "contribution" from different modes
    
plt.figure()
plt.plot(np.cumsum(-np.sort(-np.ravel(energies[0,:]))), marker='.')

# %% Plot angles

plt.figure()
plt.plot((np.unwrap(np.angle(eigamps[0,:,3]))-np.unwrap(np.angle(eigamps[0,:,5]))*2)/np.pi)
plt.plot((np.unwrap(np.angle(eigamps[2,:,1]))*1-np.unwrap(np.angle(eigamps[0,:,5]))*3)/np.pi)
"""

# %% Plot vs. unmodified frequencies

plt.figure()
kxp, kyp = np.meshgrid(np.array(range(8), dtype=np.int32), np.array(range(1,6), dtype=np.int32))
plt.scatter(kxp, kyp)
numer = kyp
denom = kyp**2 + kxp**2
gcd = np.gcd(numer, denom)
rednum = numer//gcd
redden = denom//gcd

for i in range(numer.shape[0]):
    for j in range(numer.shape[1]):
        plt.text(kxp[i,j]+0.1, kyp[i,j]+0.1, '{}/{}'.format(str(rednum[i,j]), str(redden[i,j])))

plt.xlabel('kx [plasma]')
plt.ylabel('ky [plasma]')

for i in range(10):
    eig = inds[i] % 2048
    ky = inds[i] // 2048 + 1
    eigfft = np.fft.rfft(np.real(eigs[ky-1]['vl'][:,eig]))
    kx = np.argmax(np.abs(eigfft))
    angle = np.angle(eigfft[kx])
    plt.text(kx-0.1, ky-angle/20.0, str(i+1))

# %% Check how the action norm is defined

kd = 0
ky = 3
lap = eigsolver.cent_d2x - np.eye(eigsolver.nx)*(ky**2 + kd**2)
invlap = np.linalg.inv(lap)
localfreq = - ky * np.diag(np.sqrt(-eigsolver.b)) @ invlap @ np.diag(np.sqrt(-eigsolver.b))

energy1 = np.dot(eigs[ky-1]['vpsi'][:,eig], eigs[ky-1]['vr'][:,eig])
energy2 = np.dot(eigs[ky-1]['vh'][:,eig], localfreq @ eigs[ky-1]['vh'][:,eig])

print(energy1)
print(energy2)

# %% Frequency errors

fracpart = np.zeros(9)

def l1_dev(basefreq):
    totaldev = 0.0
    
    for i in range(len(fracpart)):
        eig = inds[i] % 2048
        ky = inds[i] // 2048 + 1
        expfreq = np.imag(dmdfreqs[ky-1,eig])
        amp = actions[ky-1,eig] / ky
        fracparta = (expfreq / basefreq - np.round(expfreq / basefreq)) * basefreq
        
        totaldev = totaldev + np.abs(fracparta) * amp
        #totaldev = max((totaldev, np.abs(fracparta)))
        
    return totaldev


basefreqres = scipy.optimize.minimize_scalar(l1_dev, bounds=(-0.90/2.0, -0.78/2.0), method='bounded')
#basefreqres = scipy.optimize.minimize_scalar(l1_dev, bounds=(-0.13, -0.11), method='bounded')
basefreq = basefreqres.x
print(basefreqres)

#pos = -1.0*np.logspace(-2.0, -0.0, num=2048)
#devs = list(map(l1_dev, pos))
#plt.figure()
#plt.scatter(-pos, devs)
#plt.xscale('log')
#plt.yscale('log')

plt.figure()
for i in range(len(fracpart)):
    eig = inds[i] % 2048
    ky = inds[i] // 2048 + 1

    expfreq=np.average(np.diff(np.unwrap(np.angle(eigamps[ky-1,:,eig]))))/dt
    
    dmdfreq = dmdfreqs[ky-1,eig]
    #ax[j,k+1].axvline(np.real(dmdfreq), ls='--', c='C1')
    #ax[j,k+1].axvline(np.imag(dmdfreq), ls='--', c='C3')
    
    expfreq = np.imag(dmdfreq)
    
    fracpart[i] = (expfreq / basefreq - np.round(expfreq / basefreq)) * basefreq
    plt.plot([i,i], [0,fracpart[i]], c='C0')
    plt.scatter(i, fracpart[i], c='C0')
    plt.text(i, fracpart[i], str(int(np.round(expfreq / basefreq))))
    
plt.axhline(ls='--', c='k')
plt.axhline(np.abs(basefreq)/2.0, ls='--', c='k')
plt.axhline(-np.abs(basefreq)/2.0, ls='--', c='k')

# %%

psivfft = np.zeros((2048, 1025), dtype=complex)
psivfft[:,3] = eigs[2]['vpsi'][:,1]
psiv = np.fft.irfft(psivfft, axis=1)
plt.figure()
plt.imshow(psiv, origin='lower')

# %%

eig = 11
plt.figure()
plt.plot(np.real(eigs[1]['vr'][:,eig]))