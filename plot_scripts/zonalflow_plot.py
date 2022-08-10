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

def poincarePlot(ax, pdata, colordata):
    z0 = pdata['y'][:,0]
    yclip = pdata['yclip']
    
    nparticles = len(z0)//2
    colors = np.zeros((nparticles, yclip.shape[1]))
    colors[:,:] = colordata[:, np.newaxis]
    ax.set_aspect('equal', adjustable='datalim')

    #ax.set_xticks([])
    sc = ax.scatter(yclip[nparticles:,:], yclip[:nparticles,:], s=(72.0/450.0)**2, marker='o', linewidths=0, c=colors[:,:], rasterized=True, cmap='viridis', vmin=1.0, vmax=2.0)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(sc, cax=cax)
    
    ax.set_xlim([-np.pi,np.pi])
    ax.set_ylim([-np.pi,np.pi])
    
    #ax.set_title('$d_{cor}$')
    
    cax.text(0.5, 1.05, '$d_C$', transform=cax.transAxes, ha='center', va='bottom')
    

def zonalPlot(axm, case):
    mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))
    qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))
    
    inputdata = np.load('../poincare_input/case{}_poincare_config_fd_smooth_uphavg.npz'.format(case))
    
    xsort = np.argsort(np.ravel(mdata['allxavgs']))
    xorbit = np.average(np.reshape(np.ravel(mdata['allxavgs'])[xsort], mdata['allxavgs'].shape), axis=1)
    chfraction = np.average(np.reshape(np.ravel(mdata['allcorrdims'])[xsort]>1.5, mdata['allxavgs'].shape), axis=1)
    
    
    
    # Compute things to plot
    uy = inputdata['uy']
    
    # Plot of mixing
    axt = axm.twinx()
    
    axt.axhline(0, c='k', ls='--', lw=0.3)
    axt.plot(x, uy, c='tab:purple')
    
    axm.spines['left'].set_color('tab:purple')
    axt.spines['left'].set_color('tab:purple')
    
    axm.spines['right'].set_color('tab:green')
    axt.spines['right'].set_color('tab:green')
    
    axt.yaxis.label.set_color('tab:purple')
    axt.tick_params(axis='y', colors='tab:purple')
    
    axm.yaxis.label.set_color('tab:green')
    axm.tick_params(axis='y', colors='tab:green')
    
    uylim = np.max(np.abs(uy))
    axt.set_ylim([-uylim*1.05, uylim*1.05])
    
    if (case==1):
        axm.text(0.0, 1.02, r"$U(y)$", transform=axm.transAxes, ha='left', va='bottom', c='tab:purple')
        axm.text(1.0, 1.02, r'$\tilde{q}_{eig}$', transform=axm.transAxes, ha='right', va='bottom', c='tab:green')
    
    axm.yaxis.tick_right()
    axt.yaxis.tick_left()
    
    axm.set_xlim([-np.pi, np.pi])
    
    """
    ### Plot chaotic regions as spans ###
    
    chaotic = chfraction > 0.5
    chinds = np.where(np.diff(chaotic))[0]
    print(chinds)
    for i in range(len(chinds)//2):
        axm.axvspan(xorbit[chinds[2*i]], xorbit[chinds[2*i+1]], fc=mpl.cm.tab20(0.175))
    """
    
    ### Plotting for qtilde ###

    # Compute q for each eigenfunction    
    numeigs = inputdata['psiv'].shape[0]
    psiv = inputdata['psiv']
    kys = inputdata['kys']
    qv = np.zeros(inputdata['psiv'].shape)
    dx = 2*np.pi/nx
    cent_d2x = (np.diag(np.ones(nx-1), 1)+np.diag(np.ones(nx-1), -1) - 2*np.eye(nx) + np.diag(np.ones(1), -(nx-1))+np.diag(np.ones(1), (nx-1))) / dx**2

    for i in range(numeigs):
        ky = kys[i]
        lap = (cent_d2x - np.eye(nx)*(ky**2))
        qv[i,:] = lap @ psiv[i,:]
    
    # Compute qtilde
    camp = inputdata['amps']
    plotind = np.argmax(camp)
    
    qtildeplot = camp[:, np.newaxis] * qv
    
    # Plot qtilde
    
    axm.plot(x, qtildeplot[plotind,:], c='tab:green')
    
    qlim = np.max(np.abs(qtildeplot[plotind,:]))
    
    axm.set_ylim([-qlim*1.05, qlim*1.05])

# %% Plot figure

#fig = plt.figure(figsize=(17.8/2.54, 0.32*17.8/2.54), dpi=300)
#gs = fig.add_gridspec(1, 3, width_ratios=[2,2,2])


fig = plt.figure(figsize=(8.7/2.54, 0.7*8.7/2.54), dpi=300)
gs = fig.add_gridspec(2, 1)

#case = 1

axm1 = fig.add_subplot(gs[0])
zonalPlot(axm1, 1)
axm2 = fig.add_subplot(gs[1])
zonalPlot(axm2, 2)








plt.tight_layout(h_pad=0.0, w_pad=1.5)
plt.tight_layout(h_pad=0.0, w_pad=1.5)
plt.margins(0, tight=True)

#plt.savefig('mixing_plot.pdf', dpi=300)
#plt.savefig('mixing_plot.png', dpi=300)