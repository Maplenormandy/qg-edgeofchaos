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

ampfile1 = np.load('../dns_input/case1/eigencomps_fd_qbar.npz')
eigamps1 = ampfile1['amps']
qbar1 = ampfile1['qbar']
podsvals1 = np.loadtxt('../dns_input/case1/podsvals.txt')

dt = 0.25


eigsolver = EigenvalueSolverFD(qbar1)
nky = 8
eigs1 = [None]*nky
for ky in range(1,nky+1):
    print(ky)
    try:
        eigs1[ky-1] = np.load('../scratch/case{}_eigsolver_ky{}.npz'.format(1, ky))
        print("Loaded")
    except:
        print("Solving")
        eigs1[ky-1] = eigsolver.solveEigenfunctions(ky=ky, norm='action')
    

ampfile2 = np.load('../dns_input/case2/eigencomps_fd_qbar.npz')
eigamps2 = ampfile2['amps']
qbar2 = ampfile2['qbar']
podsvals2 = np.loadtxt('../dns_input/case2/podsvals.txt')

dt = 0.25


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


fig = plt.figure(figsize=(17.8/2.54, 0.5*11.4/2.54), dpi=300)
gs = fig.add_gridspec(2, 4, width_ratios=[2,2,2,2])


kys = [1, 3]
eignums = [1, 1]



for i in range(2):
    gs1 = gs[i,0].subgridspec(1,2, wspace=0.0)
    gs2 = gs[i,1].subgridspec(1,2, wspace=0.0)
    
    #ax = [fig.add_subplot(gs[i,j]) for j in range(4)]
    gss = [gs1[0], gs1[1], gs2[0], gs2[1]]
    ax = [fig.add_subplot(gsi) for gsi in gss]
    
    ky = kys[i]
    eig = eignums[i]
    
    # Add the plot titles as "superplots"
    if i == 0:
        axg0 = fig.add_subplot(gs[i,0], frameon=False)
        axg1 = fig.add_subplot(gs[i,1], frameon=False)
        
        axg0.set_title('POD modes')
        axg1.set_title('Eigenmode')
        axg0.xaxis.set_ticks([])
        axg0.yaxis.set_ticks([])
        axg1.xaxis.set_ticks([])
        axg1.yaxis.set_ticks([])
    
    # Pick out the right values based on which case we're looking at
    if i == 0:
        eigs = eigs1
        qbar = qbar1
        
        pod_re = np.load('../dns_input/case1/raw_podmode001.npy')
        pod_im = np.load('../dns_input/case1/raw_podmode002.npy')
        
        pod_zf = np.load('../dns_input/case1/raw_podmode000.npy')
        
        timetraces = np.load('../dns_input/case1/pod_timetraces.npz')
        pod_amp = timetraces['arr_0'][:,1] + 1j*timetraces['arr_0'][:,2]
        podsvals = (podsvals1[1], podsvals1[2])
        eigamps = eigamps1
        
        zf_amp = np.average(timetraces['arr_0'][:,0]*podsvals1[0])
        
        rsquareds = np.ravel(rsquareds1)
        vphs = np.ravel(np.array([eigs[ky-1]['w'] for ky in range(1,len(eigs)+1)]))
    else:
        eigs = eigs2
        qbar = qbar2
        
        pod_re = np.load('../dns_input/case2/raw_podmode005.npy')
        pod_im = np.load('../dns_input/case2/raw_podmode006.npy')
        
        pod_zf = np.load('../dns_input/case2/raw_podmode000.npy')
        
        timetraces = np.load('../dns_input/case2/pod_timetraces.npz')
        pod_amp = timetraces['arr_0'][:,5] + 1j*timetraces['arr_0'][:,6]
        podsvals = (podsvals1[5], podsvals1[6])
        eigamps = eigamps2
        
        zf_amp = np.average(timetraces['arr_0'][:,0]*podsvals2[0])
        
        rsquareds = np.ravel(rsquareds2)
        vphs = np.ravel(np.array([eigs[ky-1]['w'] for ky in range(1,len(eigs)+1)]))


    # POD plots
    ax[0].imshow(np.fliplr(pod_re), origin='lower')
    ax[1].imshow(np.fliplr(pod_im), origin='lower')
    
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    
    
    psivfft = np.zeros((2048, 1025), dtype=complex)
    psivfft[:,ky] = eigs[ky-1]['vpsi'][:,eig]
    psiv = np.fft.irfft(psivfft, axis=1)
    
    # Eigenfunction plots
    ax[2].imshow(np.fliplr(psiv), origin='lower')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    
    # Computing frequencies
    t = np.linspace(0, 64, num=256, endpoint=True)
    fit = Polynomial.fit(t,np.unwrap(np.angle(pod_amp)), deg=1).convert()
    podfreq = fit.coef[1]
    eigfreq = eigs[ky-1]['w'][eig]*ky
    
    plt.tight_layout()
    
    lines = ['$k_x = {}, k_y = 1$'.format(ky),
             '$\\omega_{eig} \\approx ' + str(np.round(eigfreq, 2)) + '$',
             '$\\omega_{POD} \\approx ' + str(np.round(podfreq, 2)) + '$',
             '$\\omega_{0,k} = ' + str(-8.0 * ky / (1 + ky**2)) + '$']
    
    # Eigenfunction radial plots
    ax[3].axis('off')
    ax[3].text(0.5, 0.5, '\n'.join(lines), ha='center', va='center', transform=ax[3].transAxes)
    
    
    gs_inner = gs[i,2].subgridspec(2, 1, hspace=0.0)
    
    axi0 = fig.add_subplot(gs_inner[0])
    axi1 = fig.add_subplot(gs_inner[1])
    
    nx = 2048
    x = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)
    
    time_amp = np.sqrt(np.average(np.abs(pod_amp)**2))
    #print(np.max(pod_re)*time_amp*podsvals[0])
    
    pod_fft = np.fft.fft(pod_re-1j*pod_im, axis=1)
    pod_radial_raw = pod_fft[:,ky]
    
    leftover_power = lambda theta: np.sum(np.imag(pod_radial_raw*np.exp(1j*theta))**2)
    res = scipy.optimize.minimize(leftover_power, 0.0, bounds=[(-np.pi, np.pi)])
    
    time_eigamp = np.sqrt(np.average(np.abs(eigamps[ky-1,:,eig])**2))
    
    pod_psi = np.real(pod_radial_raw * np.exp(1j*res.x) * time_amp * np.sqrt((podsvals[0]**2 + podsvals[1]**2)/2.0)) / 2048
    eig_psi = eigs[ky-1]['vpsi'][:,eig] * time_eigamp / 1024
    
    kx = np.fft.rfftfreq(nx, d=1.0/nx)
    pod_q = np.fft.irfft(-np.fft.rfft(pod_psi)*kx**2)
    eig_q = np.fft.irfft(-np.fft.rfft(eig_psi)*kx**2)
    
    pod_e = np.sum(-pod_psi*pod_q)
    eig_e = np.sum(-eig_psi*eig_q)
    
    axi0.plot(x, pod_q, lw=0.8)
    axi0.plot(x, eig_q, lw=0.8, ls='--')
    axi0.xaxis.set_ticklabels([])
    
    psi_zf = np.average(pod_zf, axis=1)*zf_amp
    q_zf = np.fft.irfft(-np.fft.rfft(psi_zf)*kx**2)
    
    #axi1.plot(x, q_zf, lw=0.8)
    #axi1.plot(x, qbar, lw=0.8, ls='--')
    
    axi1.plot(x, qbar, lw=0.8)
    
    axi0.text(0.03, 0.95, r'$\tilde{q}$', transform=axi0.transAxes, ha='left', va='top')
    axi1.text(0.03, 0.95, r'$\bar{q}$', transform=axi1.transAxes, ha='left', va='top')
    #axi1.set_yscale('log')
    
    # Plot of rsquareds versus vph
    axr = fig.add_subplot(gs[i,3])
    coherent = np.logical_and(vphs < 0.0, rsquareds > 0.4)
    axr.scatter(vphs[coherent], rsquareds[coherent], s=4.0, marker='^', c='tab:green')
    axr.scatter(vphs[np.logical_not(coherent)], rsquareds[np.logical_not(coherent)], s=1.0, marker='.', c='tab:red')
    axr.set_ylabel(r'$r^2_{\mathrm{min}}$')
    
    if i == 0:
        #ax[0].set_title('(a)', loc='left')
        #ax[2].set_title('(b)', loc='left')
        #axi0.set_title('(c)', loc='left')
        axi1.xaxis.set_ticklabels([])
        
        axi0.set_title('PV Comparison')
        axr.set_title('Coherent Eigenmodes')
        
        ax[0].text(0.0, 1.1, '(a)', transform=ax[0].transAxes, ha='left', va='bottom')
        ax[2].text(0.0, 1.1, '(b)', transform=ax[2].transAxes, ha='left', va='bottom')
        axi0.text(0.0, 1.1, '(c)', transform=axi0.transAxes, ha='left', va='bottom')
        axr.text(0.0, 1.1, '(d)', transform=axr.transAxes, ha='left', va='bottom')
        
        ax[0].text(0.05,0.95, r'$\psi(x,y)$', transform=ax[0].transAxes, ha='left', va='top')
        ax[0].set_ylabel('Case 1, modes 1+2')
    elif i == 1:
        axi1.set_xlabel('$y$')
        
        ax[0].set_ylabel('Case 2, modes 5+6')
        axr.set_xlabel(r'$u_{ph}$')


plt.tight_layout(w_pad=0.4, h_pad=0.4)
plt.tight_layout(w_pad=0.4, h_pad=0.4)

plt.savefig('eigencomparison_plots.pdf', dpi=1200)
plt.savefig('eigencomparison_plots.png', dpi=1200)

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
    
    