# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:35:45 2021

@author: maple
"""

import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate
import scipy.optimize
import h5py
from numpy.polynomial import Polynomial

import sys, os

os.chdir('../')
sys.path.append(os.path.abspath('../qg_dns/analysis/eigenvectors'))
from chm_utils import EigenvalueSolverFD

font = {'size'   : 6,
        'family' : 'sans-serif',
        'serif'  : ['CMU Serif'],
        'sans-serif' : ['CMU Sans Serif']}

linescale = 0.4
mpl.rc('axes', unicode_minus=False, linewidth=linescale)

tick_major = {'size': 3.5*linescale, 'width': 0.8*linescale}
tick_minor = {'size': 2.0*linescale, 'width': 0.6*linescale}

mpl.rc('xtick.major', **tick_major)
mpl.rc('ytick.major', **tick_major)
mpl.rc('xtick.minor', **tick_minor)
mpl.rc('ytick.minor', **tick_minor)
mpl.rc('xtick', direction='in')
mpl.rc('ytick', direction='in')

mpl.rc('font', **font)

mpl.rc('mathtext', fontset='cm')

from numpy.polynomial import Polynomial

# %% Solve for eigenfunctions

nx = 2048

ampfile1 = np.load('../dns_input/case1/eigencomps_fd_qbar.npz')
eigamps1 = ampfile1['amps']
qbar1 = ampfile1['qbar']
podsvals1 = np.loadtxt('../dns_input/case1/podsvals.txt')

dt = 0.25

nky = 8
utildey1 = np.zeros((nky, nx, nx))
eigsolver = EigenvalueSolverFD(qbar1)
eigs1 = [None]*nky
for ky in range(1,nky+1):
    print(ky)
    try:
        eigs1[ky-1] = np.load('../scratch/case{}_eigsolver_ky{}.npz'.format(1, ky))
        print("Loaded")
    except:
        print("Solving")
        eigs1[ky-1] = eigsolver.solveEigenfunctions(ky=ky, norm='action')
        
    psiv = eigs1[ky-1]['vpsi']
    psiv1 = np.roll(psiv, 1, axis=0)
    psiv2 = np.roll(psiv, -1, axis=0)
    
    utildey = (psiv1 - psiv2) / (2 * 2 * np.pi / nx)
    utildey1[ky-1,:,:] = utildey

ampfile2 = np.load('../dns_input/case2/eigencomps_fd_qbar.npz')
eigamps2 = ampfile2['amps']
qbar2 = ampfile2['qbar']
podsvals2 = np.loadtxt('../dns_input/case2/podsvals.txt')

dt = 0.25


utildey2 = np.zeros((nky, nx, nx))
eigsolver = EigenvalueSolverFD(qbar2)
nky = 8
eigs2 = [None]*nky
for ky in range(1,nky+1):
    print(ky)
    try:
        eigs2[ky-1] = np.load('../scratch/case{}_eigsolver_ky{}.npz'.format(2, ky))
        print("Loaded")
    except:
        print("Solving")
        eigs2[ky-1] = eigsolver.solveEigenfunctions(ky=ky, norm='action')

    psiv = eigs2[ky-1]['vpsi']
    psiv1 = np.roll(psiv, 1, axis=0)
    psiv2 = np.roll(psiv, -1, axis=0)
    
    utildey = (psiv1 - psiv2) / (2 * 2 * np.pi / nx)
    utildey2[ky-1,:,:] = utildey
    
# %% Get rsquared for the eigenmodes

def rsquareds(eigamps):
    numofs = 64
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
    return rsquaredall

rsquareds1 = rsquareds(eigamps1)
rsquareds2 = rsquareds(eigamps2)


# %%


fig = plt.figure(figsize=(8.7/2.54, 0.7*8.7/2.54), dpi=300)
gs = fig.add_gridspec(2, 2)


for i in range(2):
    # Pick out the right values based on which case we're looking at
    if i == 0:
        eigs = eigs1
        qbar = qbar1
        
        
        pod_zf = np.load('../dns_input/case1/raw_podmode000.npy')
        
        timetraces = np.load('../dns_input/case1/pod_timetraces.npz')
        pod_amp = timetraces['arr_0'][:,1] + 1j*timetraces['arr_0'][:,2]
        podsvals = (podsvals1[1], podsvals1[2])
        eigamps = eigamps1
        
        zf_amp = np.average(timetraces['arr_0'][:,0]*podsvals1[0])
        
        rsquareds = np.ravel(rsquareds1)
        vphs = np.ravel(np.array([eigs[ky-1]['w'] for ky in range(1,len(eigs)+1)]))
        
        utildey = utildey1
    else:
        eigs = eigs2
        qbar = qbar2
        
        
        pod_zf = np.load('../dns_input/case2/raw_podmode000.npy')
        
        timetraces = np.load('../dns_input/case2/pod_timetraces.npz')
        pod_amp = timetraces['arr_0'][:,5] + 1j*timetraces['arr_0'][:,6]
        podsvals = (podsvals1[5], podsvals1[6])
        eigamps = eigamps2
        
        zf_amp = np.average(timetraces['arr_0'][:,0]*podsvals2[0])
        
        rsquareds = np.ravel(rsquareds2)
        vphs = np.ravel(np.array([eigs[ky-1]['w'] for ky in range(1,len(eigs)+1)]))
        
        utildey = utildey2

    uwavemax = np.zeros((nky, nx))
    
    ### Compute the maximum velocity for each mode ###
    for ky in range(1,nky+1):
        
        
        uwavemax[ky-1,:] = np.max(np.abs(utildey[ky-1,:,:]), axis=0) * np.max(np.abs(eigamps[ky-1,:,:]), axis=0) / 1024
        #uwavemax[ky-1,:] = np.max(np.abs(eigamps[ky-1,:,:]), axis=0) / 1024
        
    uwavemaxr = np.ravel(uwavemax)
    
    
    kx = np.fft.rfftfreq(nx, d=1.0/nx)
    psi_zf = np.average(pod_zf, axis=1)*zf_amp
    u_zf = np.fft.irfft(-1j*np.fft.rfft(psi_zf)*kx)
        
    ### Coherence plot ###
    
    if i == 0:
        axr = fig.add_subplot(gs[0,i])
    else:
        axr = fig.add_subplot(gs[0,i], sharex=axr)
        
        
    axr.axvspan(np.min(u_zf), np.max(u_zf), zorder=0, fc=mpl.cm.tab20(7/20.0 + 1.0/40.0))
    
    coherent = np.logical_and(vphs < 0.0, rsquareds > 0.4)
    axr.scatter(vphs[coherent], rsquareds[coherent], s=4.0, marker='^', c='tab:green')
    axr.scatter(vphs[np.logical_not(coherent)], rsquareds[np.logical_not(coherent)], s=1.0, marker='.', c='tab:red', rasterized=True)
    axr.set_ylabel(r'$r^2_{\mathrm{min}}$')
    
    ### Amplitude plot ###
    
    axa = fig.add_subplot(gs[1,i], sharex=axr)
    
    axa.axvspan(np.min(u_zf), np.max(u_zf), zorder=0, fc=mpl.cm.tab20(7/20.0 + 1.0/40.0))
    
    axa.scatter(vphs[coherent], uwavemaxr[coherent], s=4.0, marker='^', c='tab:green')
    axa.scatter(vphs[np.logical_not(coherent)], uwavemaxr[np.logical_not(coherent)], s=1.0, marker='.', c='tab:red', rasterized=True)
    
    axa.set_ylabel(r'$\mathrm{sup}|u_{\mathrm{wave}}|$')
    
    print(np.max(-uwavemaxr[coherent]/vphs[coherent]))
    
    axr.set_ylim([-0.03, 1.05])
    
    
    axa.xaxis.grid(which='major', ls=':', lw=0.4)
    axr.xaxis.grid(which='major', ls=':', lw=0.4)
    axa.yaxis.grid(which='major', ls=':', lw=0.4)
    axr.yaxis.grid(which='major', ls=':', lw=0.4)
    
    [axa.spines[s].set_visible(False) for s in axa.spines]
    [axr.spines[s].set_visible(False) for s in axr.spines]
    
    #axa.text(0.95, 0.95, 'Case '+str(i+1), ha='right', va='top', transform=axa.transAxes)
    #axr.text(0.95, 0.95, 'Case '+str(i+1), ha='right', va='top', transform=axr.transAxes)
    
    if i == 0:
        axa.set_ylim([-0.035, 1.15])
        
        #plt.setp(axr.get_xticklabels(), visible=False)
        #plt.setp(axa.get_xticklabels(), visible=False)
        
        #axr.set_title('Coherence')
        #axa.set_title('Amplitude')
        
        axr.text(-0.1, 1.00, '(a)', transform=axr.transAxes, ha='right', va='bottom')
        axa.text(-0.1, 1.00, '(b)', transform=axa.transAxes, ha='right', va='bottom')
        
        axr.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=2.0))
        
    elif i == 1:
        axa.set_ylim([-0.01, 0.35])
        
        #axr.set_xlabel(r'$u_{ph}$')
    
    axa.set_xlabel(r'$u_{ph}$')    
    axr.set_title('Case {}'.format(i+1))

plt.tight_layout(pad=0, w_pad=1.2, h_pad=0.4)
plt.tight_layout(pad=0, w_pad=1.2, h_pad=0.4)
#plt.margins(0, tight=True)

plt.savefig('./final/plot_coherence.pdf', dpi=300)
plt.savefig('./final/plot_coherence.png', dpi=600)

"""
ax0 = fig.add_subplot(gs[i, 0])
ax0.axis('off')

ax1 = fig.add_subplot(gs[i, 1])
ax1.imshow(np.fliplr(poddata), origin='lower')
ax1.set_xticks([])
ax1.set_yticks([])

ax2 = fig.add_subplot(gs[i, 2])

ky = kys[i]
eig = eignums[i]


"""
