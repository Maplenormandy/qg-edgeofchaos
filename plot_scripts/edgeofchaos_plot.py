# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:53:18 2022

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


# %% Helper functions

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
    
# %% Setting up plot


fig = plt.figure(figsize=(8.7/2.54, 0.7*8.7/2.54), dpi=300)
#fig = plt.figure(figsize=(17.8/2.54, 0.25*17.8/2.54), dpi=300)
gs = fig.add_gridspec(2, 2, width_ratios=[2,2], height_ratios=[2,5])

for case in [1,2]:
    #cdata = np.load('case{}_snapcontours.npz'.format(case))
    mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))
    qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))
    
    axm = fig.add_subplot(gs[0,case-1])
    
    # Compute things to plot
    qbar = np.average(qbars['qbar'], axis=0)
    dqbar = np.gradient(qbar) / np.gradient(x) + 8
    
    xsort = np.argsort(np.ravel(mdata['allxavgs']))
    xorbit = np.average(np.reshape(np.ravel(mdata['allxavgs'])[xsort], mdata['allxavgs'].shape), axis=1)
    chfraction = np.average(np.reshape(np.ravel(mdata['allcorrdims'])[xsort]>1.5, mdata['allxavgs'].shape), axis=1)
    
    kuofraction = np.average((np.gradient(qbars['qbar'], axis=1) / (2*np.pi/nx))+8 < 0, axis=0)
    
    # Plot of mixing
    axt = axm.twinx()
    
    axm.fill_between(xorbit, chfraction, color='tab:orange', fc=mpl.cm.tab20(0.175), lw=0)
    axm.plot(xorbit, chfraction, c='tab:orange')
    axt.plot(x, dqbar, c='tab:blue')
    
    axm.set_ylim([0.0, 1.0])
    if (case==1):
        axt.set_ylim([0.0, 40.0])
    else:
        axt.set_ylim([0.0, 30.0])
        
    
    
    # Extra poincare plots
    snapind = 249 if (case == 1) else 32
    pdata = np.load('../extra_poincare_sections/case{}_section_ind{:03d}_uphavg.npz'.format(case,snapind))
    
    axp = fig.add_subplot(gs[1,case-1])
    
    poincarePlot(axp, pdata, mdata['allcorrdims'][:,snapind])
    
    if (case==1):
        axm.set_title('Case 1')
        axm.set_ylabel('$f_{chaotic}$')
        axm.set_xlabel('$y$')
        axp.set_xlabel('$x$')
        axp.set_ylabel('$y$')
    else:
        axm.set_title('Case 2')
        axt.set_ylabel(r"$\bar{q}'(y)$")
        

plt.tight_layout(h_pad=0.0, w_pad=0.0)
plt.tight_layout(h_pad=0.0, w_pad=0.0)


plt.savefig('edgeofchaos_plot.pdf', dpi=900)
plt.savefig('edgeofchaos_plot.png', dpi=900)