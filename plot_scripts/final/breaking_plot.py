# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:33:46 2022

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal
from mpl_toolkits.axes_grid1 import make_axes_locatable

import h5py

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

# %% Load eigenfunctions

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
        
        

t = np.arange(64.01, step=0.25)


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

# %% Setup helper info

def circularInterpolant(x, vec, ofs, pad):
    xp = np.zeros(len(x)+2*pad)
    xp[pad:-pad] = x
    xp[:pad] = x[-pad:] - ofs
    xp[-pad:] = x[:pad] + ofs
    
    
    vecp = np.zeros(len(x)+2*pad)
    vecp[pad:-pad] = vec
    vecp[:pad] = vec[-pad:]
    vecp[-pad:] = vec[:pad]

    return scipy.interpolate.interp1d(xp, vecp, kind='cubic')

nx = 2048
x = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)
kx = np.fft.rfftfreq(nx, 1.0/nx)
kxinv = np.zeros(kx.shape, dtype=complex)
kxinv[kx>0] = 1.0/(-1j*kx[kx>0])


def recenter(z):
    z2 = np.zeros(z.shape[0]+1)
    z2[1:-1] = (z[1:] + z[:-1]) / 2.0
    z2[0] = 2*z[0] - z[1]
    z2[-1] = 2*z[-1] - z[-2]
    return z2

    

def breakingPlot(ax, data, plotind2=2):
    tab20c = mpl.cm.get_cmap('tab20c')
    
    
    t = data['t']
    stride = 1
    if len(t)==17:
        plotind = 16
    else:
        plotind = plotind2
    
    for i in range(len(t)):
        z = data['y{}'.format(i)]
        nparticles = z.shape[0]//2
        arclength_lastbit = np.sqrt((np.mod(z[nparticles-1]-z[0]+np.pi,2*np.pi)-np.pi)**2 + (np.mod(z[-1]-z[nparticles]+np.pi,2*np.pi)-np.pi)**2)
        arclength = np.sum(np.sqrt(np.diff(z[:nparticles])**2 + np.diff(z[nparticles:])**2))+arclength_lastbit
        print(i,arclength)
    
    if 'yclip' in data:
        #yclip = data['yclip']
        yclip0 = data['yclip'][:,0]
        yclipp = data['yclip'][:,plotind]
    else:
        yclip0 = data['yclip0']
        yclipp = data['yclip{}'.format(plotind)]
        
    nparticles0 = yclip0.shape[0]//2
    nparticlesp = yclipp.shape[0]//2
    
    
    ycenter = (np.max(yclip0[:nparticles0:stride]) - np.min(yclip0[:nparticles0:stride]))

    chop = np.abs(np.diff(yclipp[nparticlesp::stride])) > 1.5*np.pi
    chopargs = np.argwhere(chop)[:,0] + 1
    
    c2 = tab20c(4.5/20.0)
    
    ax.plot(yclipp[nparticlesp:nparticlesp+chopargs[0]:stride], yclipp[:chopargs[0]:stride], c=c2, lw=1.0)
    ax.plot(yclipp[nparticlesp+chopargs[-1]::stride], yclipp[chopargs[-1]:nparticlesp:stride], c=c2, lw=1.0)
    
    for i in range(len(chopargs)-1):
        ax.plot(yclipp[nparticlesp+chopargs[i]:nparticlesp+chopargs[i+1]:stride], yclipp[chopargs[i]:chopargs[i+1]:stride], c=c2, lw=0.8)
        
    
    ax.plot(yclip0[nparticles0::stride], yclip0[:nparticles0:stride], c=tab20c(7.5/20.0), lw=0.8)
    
    #ax.set_aspect('equal', adjustable='datalim')
    
    #ax.set_xlim([-np.pi, np.pi])
    #ax.set_ylim([-np.pi, np.pi])

# %% Plot figure

fig = plt.figure(figsize=(13.0/2.54, 13.0/2.54*0.36), dpi=300)
gs = fig.add_gridspec(1, 7, width_ratios=[2.5,1,0.2, 1, 2.5,1,0.2], wspace=0.0)


for case in [1,2]:
    cdata = np.load('case{}_snapcontours.npz'.format(case))
    mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))
    qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))
    
    ### Load the PV data ###
    #snapind = 51 if (case==1) else 192
    snapind = 0 if (case==1) else 192
    snapfilenum = (snapind//16+2) if (case == 1) else (snapind//10+1)
    simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, snapfilenum), 'r')
    qindex = (snapind%16) if (case == 1) else (snapind%10)
    
    q = simdata['tasks/q'][qindex,:,:]
    
    
    lenq = circularInterpolant(cdata['levels'], np.average(cdata['lenmaxcontour'], axis=0), 2*8*np.pi, 500)
    
    axq = fig.add_subplot(gs[4*case-4])
    
    #axq.set_yticklabels([])
    
    #axq.set_title('DNS Snapshot')
    #axq.text(0.0, 1.05, '(b)', transform=axq.transAxes, ha='left', va='bottom')
    
    axq.set_xlabel('$x$')
    axq.set_ylabel('$y$')
    
    #axq.set_xlim([-np.pi, np.pi])
    
    
    
    ### Wave breaking ###
    for suffix in ['qmin', 'qmax']:
        ldata = np.load('../sections/case{}_breaking_{}.npz'.format(case, suffix))
        plotind2 = 2 if (case==1) else 4
        breakingPlot(axq, ldata, plotind2=plotind2)
    
    
    im = axq.imshow(np.fliplr(lenq(q+8*x[:,np.newaxis])), origin='lower', cmap='viridis', extent=(-np.pi, np.pi, -np.pi, np.pi))
    
    ### Set up the eigenfunction info ###
    ky=1
    if case == 1:
        eigs = eigs1
        qbar = qbar1
        eigamps = eigamps1
        eig = [0,2]
        rsquareds = rsquareds1
        
    else:
        eigs = eigs2
        qbar = qbar2
        eigamps = eigamps2
        eig = [3,4]    
        rsquareds = rsquareds2
        
        
    ### Compute qtilde ###
    vphs = np.array([eigs[ky-1]['w'] for ky in range(1,len(eigs)+1)])
    coherent = np.logical_and(vphs < 0.0, rsquareds > 0.4)
    
    qtildesq = np.zeros(nx)
    for ky in range(1,len(eigs)+1):
        #qtilde = eigs[ky-1]['vr'] * coherent[ky-1,np.newaxis,:] * eigamps[ky-1,snapind,np.newaxis,:] / 1024 * np.sqrt(rsquareds[ky-1,np.newaxis,:])
        qtilde = eigs[ky-1]['vr'] * coherent[ky-1,np.newaxis,:] * eigamps[ky-1,snapind,np.newaxis,:] / 1024
        
        qtildesq += np.sum(np.abs(qtilde)**2, axis=1)/2
    
    ### qtilde plotting ###
    axr = fig.add_subplot(gs[4*case-3])
    
    qplot = np.array([np.zeros(nx), qtildesq])
    xplot = np.array([x, x])
    
    qbar = np.average(qbars['qbar'], axis=0) + 8*x
    cplot = np.array([lenq(qbar), lenq(qbar)])
    
    axr.pcolormesh(qplot, xplot, cplot, shading='gouraud')
    
    axr.plot(qtildesq, x, c='k', lw=0.4)
    axr.set_ylim([-np.pi, np.pi])
    
    axr.set_xlabel(r'$\langle \tilde{q}^2_{coherent} \rangle$')
    axr.set_yticklabels([])
    
    ### Decorations ###
    axq.set_title('Case {}'.format(case))
    
    #divider = make_axes_locatable(axr)
    #cax = divider.append_axes("right", size="10%", pad=0.05)
    cax = fig.add_subplot(gs[4*case-2])
    
    plt.colorbar(im, cax=cax)
    
    cax.text(0.5, 1.03, '$\ell_q$', transform=cax.transAxes, ha='center', va='bottom')
    
    axq.set_ylim([-np.pi, np.pi])
    
plt.tight_layout(pad=0, w_pad=0.0)
plt.tight_layout(pad=0, w_pad=0.0)

plt.savefig('plot_breaking.pdf', dpi=300)

# %% 