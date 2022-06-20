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


def recenter(z):
    z2 = np.zeros(z.shape[0]+1)
    z2[1:-1] = (z[1:] + z[:-1]) / 2.0
    z2[0] = 2*z[0] - z[1]
    z2[-1] = 2*z[-1] - z[-2]
    return z2

def poincarePlot(ax, pdata, colordata):
    z0 = pdata['y'][:,0]
    yclip = pdata['yclip']
    
    nparticles = len(z0)//2
    colors = np.zeros((nparticles, yclip.shape[1]))
    colors[:,:] = colordata[:, np.newaxis]
    ax.set_aspect('equal', adjustable='datalim')

    #ax.set_xticks([])
    sc = ax.scatter(yclip[nparticles:,:], yclip[:nparticles,:], s=(72.0/900.0)**2, marker='o', linewidths=0, c=colors[:,:], rasterized=True, cmap='viridis', vmin=1.0, vmax=2.0)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(sc, cax=cax)
    
    ax.set_xlim([-np.pi,np.pi])
    ax.set_ylim([-np.pi,np.pi])
    
    #ax.set_title('$d_{cor}$')
    
    cax.text(0.5, 1.05, '$d_C$', transform=cax.transAxes, ha='center', va='bottom')
    

# %% Plot figure


fig = plt.figure(figsize=(8.7/2.54, 0.5*8.7/2.54), dpi=300)

gs = fig.add_gridspec(1, 3, width_ratios=[.95,.95,.05])

for case in [1,2]:
    cdata = np.load('case{}_snapcontours.npz'.format(case))
    mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))
    qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))
    
    ### Chaotic fraction vs space-time ###
    axf = fig.add_subplot(gs[case-1])
    
    xsort = np.argsort(mdata['allxavgs'], axis=0)
    xplot = np.take_along_axis(mdata['allxavgs'], xsort, axis=0)
    corrplot = np.take_along_axis(mdata['allcorrdims'], xsort, axis=0)
    trange = np.arange(0, 64.1, 0.25)
    trange2 = np.arange(-0.125, 64.0 + 0.125 + 0.01, 0.25)
   
    for i in range(len(trange)):
        xplot2 = recenter(xplot[:,i])
        tplot2 = np.array([trange2[i], trange2[i+1]])
        
        tplot3, xplot3 = np.meshgrid(tplot2, xplot2)
        
        c = axf.pcolormesh(tplot3, xplot3, np.array([corrplot[:,i]]).T, vmin=1.0, vmax=2.0, shading='flat', rasterized=True)
        
    axf.set_ylim([-np.pi, np.pi])
    if (case==1):
        axf.set_ylabel('$y$')
        axf.set_xlabel('$t$')
        axf.set_title('Case 1')
    else:
        axf.set_title('Case 2')
        
        cax = fig.add_subplot(gs[2])
        plt.colorbar(c, cax=cax)
        
        cax.text(0.5, 1.05, '$d_C$', transform=cax.transAxes, ha='center', va='bottom')
    
plt.tight_layout(h_pad=0.0, w_pad=0.0)
plt.tight_layout(h_pad=0.0, w_pad=0.0)

plt.savefig('chaoshovmoller_plot.pdf', dpi=300)
plt.savefig('chaoshovmoller_plot.png', dpi=300)