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


# %%


fig = plt.figure(figsize=(8.7/2.54, 0.9*8.7/2.54), dpi=300)

#fig = plt.figure(figsize=(11.4/2.54, 0.45*11.4/2.54), dpi=300)
#gs = fig.add_gridspec(2, 3, width_ratios=[2,2,2])


#gsouter = fig.add_gridspec(2,1, height_ratios=[2,2], hspace=0.3)
#gs = gsouter[0].subgridspec(2,2)
#gs2o = gsouter[1].subgridspec(2,2)


cases = [1, 2, 2]
podmodes = [1, 1, 5]
kys = [1, 1, 3]
eignums = [1, 5, 1]


gs = fig.add_gridspec(len(cases),2, hspace=0.0)

case = 1

for i in range(len(cases)):
    gs1 = gs[i,0].subgridspec(1,2, wspace=0.0)
    gs2 = gs[i,1].subgridspec(1,2, wspace=0.0)
    
    #ax = [fig.add_subplot(gs[i,j]) for j in range(4)]
    gss = [gs2[0], gs2[1], gs1[1], gs1[0]]
    ax = [fig.add_subplot(gsi) for gsi in gss]
    
    case = cases[i]
    pod = podmodes[i]
    ky = kys[i]
    eig = eignums[i]
    
    # Add the plot titles as "superplots"
    if i == 0:
        axg0 = fig.add_subplot(gs[i,0], frameon=False)
        axg1 = fig.add_subplot(gs[i,1], frameon=False)
        
        axg1.set_title('POD modes')
        ax[2].set_title('Eigenmode')
        axg0.xaxis.set_ticks([])
        axg0.yaxis.set_ticks([])
        axg1.xaxis.set_ticks([])
        axg1.yaxis.set_ticks([])
    
    if case == 1:
        eigs = eigs1
        qbar = qbar1
        
        pod_re = np.load('../dns_input/case1/raw_podmode00{}.npy'.format(pod))
        pod_im = np.load('../dns_input/case1/raw_podmode00{}.npy'.format(pod+1))
        
        timetraces = np.load('../dns_input/case1/pod_timetraces.npz')
    else:
        eigs = eigs2
        qbar = qbar2
        
        pod_re = np.load('../dns_input/case2/raw_podmode00{}.npy'.format(pod))
        pod_im = np.load('../dns_input/case2/raw_podmode00{}.npy'.format(pod+1))
        
        timetraces = np.load('../dns_input/case2/pod_timetraces.npz')
    
    
    pod_amp = timetraces['arr_0'][:,pod] + 1j*timetraces['arr_0'][:,pod+1]


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
    podfreq = -np.abs(fit.coef[1])
    eigfreq = eigs[ky-1]['w'][eig]*ky
    
    plt.tight_layout()
    
    kx = (eig+1)//2
    lines = ['Case {}, modes {}+{}'.format(case, pod, pod+1),
             '$k_x = {}, k_y = {}$'.format(ky, kx),
             '$\\omega_{eig} \\approx ' + str(np.round(eigfreq, 2)) + '$',
             '$\\omega_{POD} \\approx ' + str(np.round(podfreq, 2)) + '$',
             '$\\omega_{0,k} = ' + str(-8.0 * ky / (kx**2 + ky**2)) + '$']
    
    ax[3].axis('off')
    ax[3].text(0.5, 0.5, '\n'.join(lines), ha='center', va='center', transform=ax[3].transAxes)


#plt.tight_layout(w_pad=0.0, h_pad=0.4)
#plt.tight_layout(w_pad=0.0, h_pad=0.4)
plt.margins(0, tight=True)

#plt.savefig('eigencomparison_plots.pdf', dpi=300)
#plt.savefig('eigencomparison_plots.png', dpi=300)

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
