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
    sc = ax.scatter(yclip[nparticles:,:], yclip[:nparticles,:], s=(72.0/900.0)**2, marker='o', linewidths=0, c=colors[:,:], rasterized=True, cmap='viridis', vmin=1.0, vmax=2.0)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(sc, cax=cax)
    
    ax.set_xlim([-np.pi,np.pi])
    ax.set_ylim([-np.pi,np.pi])
    
    #ax.set_title('$d_{cor}$')
    
    cax.text(0.5, 1.05, '$d_C$', transform=cax.transAxes, ha='center', va='bottom')
    

# %% Plot figure

fig = plt.figure(figsize=(17.8/2.54, 0.25*17.8/2.54), dpi=300)
gs = fig.add_gridspec(1, 4, width_ratios=[2,2,2,2])


for case in [1,2]:
    cdata = np.load('case{}_snapcontours.npz'.format(case))
    mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))
    qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))

    ### Load and plot PV data ###
    snapind = 51 if (case == 1) else 192
    snapfilenum = (snapind//16+2) if (case == 1) else (snapind//10+1)
    simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, snapfilenum), 'r')
    qindex = (snapind%16) if (case == 1) else (snapind%10)
    
    q = simdata['tasks/q'][qindex,:,:]
    
    lenq = circularInterpolant(cdata['levels'], np.average(cdata['lenmaxcontour'], axis=0), 2*8*np.pi, 500)
    
    axq = fig.add_subplot(gs[2*case-1])
    #axq.set_xticks([])
    #axq.set_yticks([])
    

    #axq.set_title('$\ell(q(x,y,t))$')
    
    
    ### Plot the poincare plots max amplitude ###
    pdata = np.load('../extra_poincare_sections/case{}_section_ind{:03d}_uphavg.npz'.format(case,snapind))
    axp1 = fig.add_subplot(gs[2*case-2])
    
    poincarePlot(axp1, pdata, mdata['allcorrdims'][:,snapind])
    
    if (case == 1):
        #axp1.text(0.0, 1.05, 'Case 1', transform=axp1.transAxes, ha='left', va='bottom')
        axp1.set_title('Case 1', loc='left')
        axp1.set_xlabel('$x$')
        axp1.set_ylabel('$y$')
    else:
        #axp1.text(0.0, 1.05, 'Case 2', transform=axp1.transAxes, ha='left', va='bottom')
        axp1.set_title('Case 2', loc='left')
        axp1.set_yticklabels([])
        
    im = axq.imshow(np.fliplr(lenq(q+8*x[:,np.newaxis])), origin='lower', cmap='viridis', extent=(-np.pi, np.pi, -np.pi, np.pi))
    divider = make_axes_locatable(axq)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(im, cax=cax)
    
    cax.text(0.5, 1.05, '$\ell_q$', transform=cax.transAxes, ha='center', va='bottom')
    axq.set_yticklabels([])
    
    
    
plt.tight_layout(h_pad=0.0, w_pad=0.0)
plt.tight_layout(h_pad=0.0, w_pad=0.0)

plt.savefig('mixing_plot.pdf', dpi=900)
plt.savefig('mixing_plot.png', dpi=900)