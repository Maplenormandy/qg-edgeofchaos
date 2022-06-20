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
import scipy

import sys, os
sys.path.append(os.path.abspath('../qg_dns/analysis/eigenvectors'))
from chm_utils import EigenvalueSolverFD



# %% Load data

case = 2
ampfile = np.load('../dns_input/case{}/eigencomps_fd_qbar.npz'.format(case))
eigamps = ampfile['amps']
qbar = ampfile['qbar']

dt = 0.25

# %% Compute scales

podsvals = np.loadtxt('../dns_input/case{}/podsvals.txt'.format(case))
urms = np.sqrt(np.sum(podsvals[:]**2)/256/2048/2048)
krhines = np.sqrt(8.0 / 2 / urms)
print('k_rhines^2', krhines**2)
print('u_rms', urms)


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
        np.savez('case{}_eigsolver_ky{}.npz'.format(case, ky), **eigs[ky-1])
    
# %% Compute the number of non-resonant modes
    
uymin = np.min(eigsolver.uy)
for ky in range(1,nky+1):
    print(ky, np.sum(eigs[ky-1]['w'] < uymin))

# %% Energy per mode
    
eigenergies = np.zeros((nky, 2048))
for ky in range(1,len(eigs)+1):
    eigenergies[ky-1,:] = -np.sum(eigs[ky-1]['vpsi']*eigs[ky-1]['vr'], axis=0)


# %% Compute DMD frequencies of individual modes

s2 = np.sum(np.abs(eigamps[:,:-1,:])**2, axis=1)
amat = np.sum(eigamps[:,1:,:]*np.conj(eigamps[:,:-1,:]), axis=1) / s2
dmdfreqs = np.log(amat) / dt

amat_unit = amat / np.abs(amat)

fitofs = 1

residuals = (eigamps[:,:-fitofs,:] * amat_unit[:,np.newaxis,:]**fitofs) - eigamps[:,fitofs:,:]
vartot = np.average(np.abs(eigamps[:,fitofs:,:] - np.average(eigamps[:,fitofs:,:], axis=1)[:,np.newaxis,:])**2, axis=1)
varresid = np.average(np.abs(residuals)**2, axis=1)

rsquared0 = 1 - (varresid/vartot)

actions = np.average(np.abs(eigamps[:,:,:])**2, axis=1)
std_actions = np.std(np.abs(eigamps[:,:,:])**2, axis=1)

energies = np.average(np.abs(eigamps[:,:,:])**2, axis=1) * eigenergies
minenergies = np.min(np.abs(eigamps[:,:,:])**2, axis=1) * eigenergies
energyinds = np.argsort(-energies, axis=None)
minenergyinds = np.argsort(-minenergies, axis=None)

coherence = (np.average(np.abs(eigamps[:,:,:])**2, axis=1) / np.std(np.abs(eigamps[:,:,:])**2, axis=1))
coherent = np.all(np.abs(eigamps[:,:,:])**2 > np.std(np.abs(eigamps[:,:,:])**2, axis=1)[:,np.newaxis,:], axis=1)

#inds = np.argsort(-coherence, axis=None)

kyeig = np.zeros((eigamps.shape[0], eigamps.shape[2]), dtype=int)
kyeig[:] = np.arange(1,nky+1, dtype=int)[:,np.newaxis]
# This is a hacky way to estimate kx
kxeig = np.zeros((eigamps.shape[0], eigamps.shape[2]), dtype=int)
kxeig[:] = np.arange(0,eigamps.shape[2], dtype=int)[np.newaxis,:]
kxeig = (kxeig + 1) // 2

k2eig = kyeig*kyeig + kxeig*kxeig
scaleinds = np.argsort(k2eig, axis=None, kind='stable')

uphinds = np.argsort(np.array([eigs[ky-1]['w'] for ky in range(1,len(eigs)+1)]), axis=None)

numofs = 64
rsqt = np.zeros((rsquared0.shape[0], numofs, rsquared0.shape[1]))
rt = np.arange(numofs)*dt

for i in range(numofs):
    fitofs = i+1
    
    x = eigamps[:,:-fitofs,:]
    y = eigamps[:,fitofs:,:]
    
    amat = np.sum(y * np.conj(x), axis=1) / np.sum(np.abs(x)**2, axis=1)
    
    residuals = y - (x * amat[:,np.newaxis,:])
    vartot = np.average(np.abs(y)**2, axis=1)
    varresid = np.average(np.abs(residuals)**2, axis=1)
    
    rsqt[:,i,:] = 1 - (varresid/vartot)
    
rsquared = np.min(rsqt, axis=1)
rsqinds = np.argsort(-rsquared, axis=None)

inds = uphinds

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
    
    ax[j,k].set_title('ky={}, kx={}, eig={}, r2={}'.format(ky,kx,eig,np.round(rsquared[ky-1, eig],2)))
    #ax[j,k].set_title('r2={}'.format(np.round(rsquared[ky-1, eig],2)))
    #ax.set_aspect('equal')
    ax[j,k].plot(t, np.real(eigamps[ky-1,:,eig]), c='#1f77b4')
    ax[j,k].plot(t, np.imag(eigamps[ky-1,:,eig]), ls='--', c='#1f77b4')
    ax[j,k].plot(t, np.abs(eigamps[ky-1,:,eig]), c='#2ca02c')
    ax[j,k].plot(t, -np.abs(eigamps[ky-1,:,eig]), c='#2ca02c')
    
    #ax[j,k].plot(t[fitofs:], np.real(residuals[ky-1,:,eig]), c='tab:red')
    #ax[j,k].plot(t[fitofs:], np.imag(residuals[ky-1,:,eig]), c='tab:red', ls='--')
    
    
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

# %% Plot of rsquared over time
    

f, ax = plt.subplots(nrows, ncols)

for i in range(nrows*ncols):
    j = i%nrows
    k = (i//nrows)
    
    eig = inds[i] % 2048
    ky = inds[i] // 2048 + 1
    
    ax[j,k].plot(rt, rsqt[ky-1,:,eig])
    #ax[j,k].set_ylim([-0.05,1.05])
    ax[j,k].plot(t, np.ones(t.shape), c='k', ls='--')
    ax[j,k].plot(t, np.zeros(t.shape), c='k', ls='--')
    
# %% Compute autocorrelation and compare with rsquared

"""
i = 1

eig = inds[i] % 2048
ky = inds[i] // 2048 + 1

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(rt, rsqt[ky-1,:,eig])

autocorrel = scipy.signal.correlate(eigamps[ky-1,:,eig], eigamps[ky-1,:,eig], mode='full')
correllags = scipy.signal.correlation_lags(257, 257, mode='full')
#yamps1 = np.cumsum(np.abs(eigamps[ky-1,::-1,eig])**2)
#yamps = np.cumsum(np.abs(eigamps[ky-1,:,eig])**2)
ycomb = np.sum(np.abs(eigamps[ky-1,:,eig])**2)

axt = ax.twinx()

axt.plot(correllags/4.0, np.abs(autocorrel/ycomb)**2, c='tab:orange')
"""

# %% Plot of rsquared versus phase velocity index

"""
ws = np.array([eigs[ky-1]['w'] for ky in range(1,nky+1)])

fig = plt.figure()

ax = plt.subplot(211)
#plt.plot(np.ravel(rsquared)[uphinds], marker='.', ls='')
for ky in range(1,len(eigs)+1):
    plt.scatter(np.ravel(ws[ky-1,:]), np.ravel(rsquared[ky-1,:]))

ax2 = plt.subplot(212, sharex=ax)
xplot = np.linspace(-np.pi, np.pi, endpoint=False, num=2048)
ax2.plot(eigsolver.uy, xplot)
#plt.axvline(uymin)
"""

# %% Frequency plots
    
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
    ax[j,k+1].axvline(np.imag(dmdfreq), ls='--', c='tab:red')
    
    # Compare frequency vs. "bare" frequency
    kx = np.argmax(np.abs(np.fft.rfft(np.real(eigs[ky-1]['vl'][:,eig]))))
    
    dispersion = -8.0 * ky / (kx**2 + ky**2)
    ax[j,k+1].axvline(dispersion, ls='--', c='tab:purple')
        
    

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

# %% Plot vs. eigenfrequencies


# %% Plot kx/ky grid
"""
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

for i in range(nrows*ncols):
    eig = inds[i] % 2048
    ky = inds[i] // 2048 + 1
    eigfft = np.fft.rfft(np.real(eigs[ky-1]['vl'][:,eig]))
    kx = np.argmax(np.abs(eigfft))
    angle = np.angle(eigfft[kx])
    plt.text(kx-0.1, ky-angle/20.0, str(i+1))
"""
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

fracpart = np.zeros(13)

ky0 = energyinds[0] // 2048 + 1
eig0 = energyinds[0] % 2048

dopplerc = np.imag(dmdfreqs[ky0-1,eig0])/ky0
#dopplerc = 0.0

def l1_dev(basefreq):
    totaldev = 0.0
    
    for i in range(len(fracpart)):
        eig = inds[i] % 2048
        ky = inds[i] // 2048 + 1
        expfreq = np.imag(dmdfreqs[ky-1,eig]) - ky*dopplerc
        ampt = eigamps[ky-1,:,eig] * (-1)**ky
        amp = np.sqrt(np.average(np.abs(ampt)**2)) / 1024
        fracparta = (expfreq / basefreq - np.round(expfreq / basefreq)) * basefreq
        
        totaldev = totaldev + np.abs(fracparta) * (amp / ky)**2
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

tab20 = mpl.cm.get_cmap('tab20')

maxfreq = np.min(eigsolver.uy)
minfreq = -8.0

plt.plot([minfreq,maxfreq],[minfreq,maxfreq], ls='--', c='k')


#plt.figure()
for i in range(len(fracpart)):
    eig = inds[i] % 2048
    ky = inds[i] // 2048 + 1

    expfreq=np.average(np.diff(np.unwrap(np.angle(eigamps[ky-1,:,eig]))))/dt
    
    dmdfreq = np.imag(dmdfreqs[ky-1,eig])
    
    #ax[j,k+1].axvline(np.real(dmdfreq), ls='--', c='C1')
    #ax[j,k+1].axvline(np.imag(dmdfreq), ls='--', c='C3')
    
    expfreq = dmdfreq - ky*dopplerc
    
    eigfreq = eigs[ky-1]['w'][eig]
    
    color = tab20((2*(ky-1) + (1-eig%2))/20.0 + 0.025)
    
    plt.scatter(eigfreq, dmdfreq/ky, color=color, marker='.')
    plt.scatter(eigfreq, np.round(expfreq / basefreq)/ky*basefreq + dopplerc, color=color, marker='+')
    plt.text(eigfreq, dmdfreq/ky, '{}: ({},{})'.format(i, ky, eig))
    
    
    fracpart[i] = (expfreq / basefreq - np.round(expfreq / basefreq)) * basefreq
    #plt.plot([i,i], [0,fracpart[i]], c='C0')
    #plt.scatter(i, fracpart[i], c='C0')
    #plt.text(i, fracpart[i], str(int(np.round(expfreq / basefreq))))
    
plt.axis('square')
#plt.axhline(ls='--', c='k')
#plt.axhline(np.abs(basefreq)/2.0, ls='--', c='k')
#plt.axhline(-np.abs(basefreq)/2.0, ls='--', c='k')

# %%

"""
plt.figure()
betadivs = np.geomspace(0.06, 0.08, num=2048)
freqdevs = list(map(l1_dev, - betadivs))
plt.semilogx(betadivs, freqdevs)
plt.semilogx(betadivs, 3*betadivs)
"""

# %%

"""
ky = 1
eig = 1923
psivfft = np.zeros((2048, 1025), dtype=complex)
psivfft[:,ky] = eigs[ky-1]['vpsi'][:,eig]
psiv = np.fft.irfft(psivfft, axis=1)
plt.figure()
plt.imshow(psiv, origin='lower')
"""

# %% Plot of vph vs. eigenfunction index

"""
ky = 2
plt.figure()
plt.scatter(range(2048), np.average(np.abs(eigamps[ky-1,:,:])**2, axis=0)/np.sqrt(ky))
plt.yscale('log')
"""

# %% Eigenfunction plots

eig = 4
ky = 4
fig = plt.figure()
ax = plt.subplot(211)
ax.plot(np.real(eigs[ky-1]['vpsi'][:,eig]))
axt = ax.twinx()
axt.plot(eigsolver.uy, c='tab:orange')

ax2 = plt.subplot(212)
ax2.plot(np.real(eigamps[ky-1,:,eig]), c='tab:blue')
ax2.plot(np.imag(eigamps[ky-1,:,eig]), c='tab:blue', ls='--')

# %% Plot to check if the eigencomponent frequency varies with the amplitude

"""
i = 18

eig = inds[i] % 2048
ky = inds[i] // 2048 + 1
freqs = np.diff(np.unwrap(np.angle(eigamps[ky-1,:,eig])))/dt
amps = np.abs(eigamps[ky-1,:,eig])**2

kx = np.argmax(np.abs(np.fft.rfft(np.real(eigs[ky-1]['vl'][:,eig]))))
dispersion = -8.0 * ky / (kx**2 + ky**2)

plt.figure()
plt.scatter(amps[:-1]+amps[1:], freqs)
plt.axhline(dispersion, ls='--', c='tab:red')
plt.axhline(np.real(eigs[ky-1]['w'][eig])*ky, ls='--', c='tab:orange')
"""

# %% Plot to check the amplitude and coherence of eigenfunctions

fig = plt.figure()

ax = plt.subplot(321)
axa = plt.subplot(322)
for ky in range(1,len(eigs)+1):
    ax.scatter(eigs[ky-1]['w'], np.average(np.abs(eigamps[ky-1,:,:])**2, axis=0)/np.sqrt(ky))
    axa.scatter(np.arange(2048), np.average(np.abs(eigamps[ky-1,:,:])**2, axis=0)/np.sqrt(ky))
ax.set_yscale('log')
axa.set_yscale('log')

ax2 = plt.subplot(323, sharex=ax)
ax2a = plt.subplot(324, sharex=axa)
for ky in range(1,len(eigs)+1):
    ax2.scatter(eigs[ky-1]['w'], np.ravel(rsquared[ky-1,:]))
    ax2a.scatter(np.arange(2048), np.ravel(rsquared[ky-1,:]))
ax2.axhline(0.5, c='k', ls='--')
ax2.axhline(0.9, c='k', ls='--')
ax2a.axhline(0.5, c='k', ls='--')
ax2a.axhline(0.9, c='k', ls='--')

ax3 = plt.subplot(325, sharex=ax)
#ax3a = plt.subplot(326, sharex=ax)
ax3a = plt.subplot(326)
xplot = np.linspace(-np.pi, np.pi, endpoint=False, num=2048)
ax3.plot(eigsolver.uy, xplot)
for ky in range(1,len(eigs)+1):
    #ax3a.scatter(eigs[ky-1]['w'], np.ravel(rsquared[ky-1,:])*np.average(np.abs(eigamps[ky-1,:,:])**2, axis=0)/np.sqrt(ky))
    #ax3a.scatter(np.imag(dmdfreqs[ky-1,:])/ky, np.ravel(rsquared[ky-1,:])*np.average(np.abs(eigamps[ky-1,:,:])**2, axis=0)/np.sqrt(ky))
    #ax3a.scatter(np.imag(dmdfreqs[ky-1,:])/ky, np.ravel(rsquared[ky-1,:]))
    ax3a.scatter(np.average(np.abs(eigamps[ky-1,:,:])**2, axis=0)/np.sqrt(ky), np.ravel(rsquared[ky-1]))
    pass
ax3a.set_xscale('log')

# %% Plot of DMD freq vs. eigenfrequency
    
plt.figure()
for ky in range(1,len(eigs)+1):
    plt.scatter(np.imag(dmdfreqs[ky-1,:])/ky, eigs[ky-1]['w'])
plt.axis('equal')

# %% Plot of the x dependence of the modes

ky = 3
plt.figure()
plt.imshow(np.clip(eigs[ky-1]['vh'], -np.sqrt(1/1024), np.sqrt(1/1024)), cmap='PiYG')
