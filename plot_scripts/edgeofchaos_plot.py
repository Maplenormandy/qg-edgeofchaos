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

# %% Compute etensors
    
etensors = [None]*2

for case in [1,2]:
    inputdata = np.load('../poincare_input/case{}_poincare_config_fd_smooth_uphavg.npz'.format(case))
    
    nx = 2048
    numeigs = inputdata['psiv'].shape[0]
    
    etensor = np.zeros((numeigs, numeigs))
    
    psiv = inputdata['psiv']
    kys = inputdata['kys']
    qv = np.zeros(inputdata['psiv'].shape)
    dx = 2*np.pi/nx
    cent_d2x = (np.diag(np.ones(nx-1), 1)+np.diag(np.ones(nx-1), -1) - 2*np.eye(nx) + np.diag(np.ones(1), -(nx-1))+np.diag(np.ones(1), (nx-1))) / dx**2
    
    for i in range(numeigs):
        ky = kys[i]
        lap = (cent_d2x - np.eye(nx)*(ky**2))
        qv[i,:] = lap @ psiv[i,:]
        
    for i in range(numeigs):
        for j in range(i,numeigs):
            if kys[i] != kys[j]:
                pass
            else:
                etensor[i,j] = -psiv[i,:] @ qv[j,:]
                etensor[j,i] = -psiv[j,:] @ qv[i,:]
                
    etensors[case-1] = etensor

# %% Setting up plot


fig = plt.figure(figsize=(17.8/2.54, 0.27*17.8/2.54), dpi=300)
#fig = plt.figure(figsize=(17.8/2.54, 0.25*17.8/2.54), dpi=300)
gs = fig.add_gridspec(1, 4)

amps = np.arange(0.1, 1.61, 0.1)

gsihist = gs[0].subgridspec(2, 1)
gsiamp = gs[1].subgridspec(2, 1)

for case in [1,2]:
    #cdata = np.load('case{}_snapcontours.npz'.format(case))
    adata = np.load('../poincare_analysis/case{}_mixing_lengths_amps.npz'.format(case))
    mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))
    
    axh = fig.add_subplot(gsihist[case-1])
    axa = fig.add_subplot(gsiamp[case-1])
    
    ### Compute chaotic fractions for snapshots ###
    chaoticfractions = np.average(mdata['allcorrdims']>1.5, axis=0)
    ampfractions = np.average((adata['allcorrdims']>1.5), axis=0)
    
    bins = np.linspace(0.0, 0.25, num=13)
    
    ### Compute average energy deviations ###
    inputdata = np.load('../poincare_input/case{}_poincare_config_fd_smooth_uphavg.npz'.format(case))
    timedata = np.load('../poincare_input/case{}_eigencomponent_timedata_uphavg.npz'.format(case))
                
    zamps = inputdata['amps']*np.exp(1j*inputdata['phases'])
    
    avgenergy = np.real(np.conj(zamps) @ (etensors[case-1] @ zamps))
    snapenergies = np.zeros(timedata['ampdevs'].shape[1])
    
    for i in range(timedata['ampdevs'].shape[1]):
        zamps = inputdata['amps']*timedata['ampdevs'][:,i]*np.exp(1j*(inputdata['phases']+timedata['phasedevs'][:,i]))
        snapenergies[i] = np.real(np.conj(zamps) @ (etensors[case-1] @ zamps))
        
    ### Plot histogram ###
    axh.hist(chaoticfractions, bins=bins)
    
    ### Plot chaotic fraction vs. amplitude ###
    axa.axvspan(np.min(np.sqrt(snapenergies/avgenergy)), np.max(np.sqrt(snapenergies/avgenergy)), fc=mpl.cm.tab20(0.175))
    #axa.axvline(1.0, c='k', ls='--')
    axa.plot(amps[:], ampfractions[:])

    
### Plot Poincare sections ###
axp1 = fig.add_subplot(gs[2])
pdata = np.load('../extra_poincare_sections/case{}_section_ind{:03d}_uphavg.npz'.format(2,0))
mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(2))

poincarePlot(axp1, pdata, mdata['allcorrdims'][:,0])

axp2 = fig.add_subplot(gs[3])
pdata = np.load('../extra_poincare_sections/case{}_section_ind{:03d}_switched_uphavg.npz'.format(2,0))
mdata = np.load('../poincare_analysis/case{}_mixing_lengths_switched.npz'.format(2))

poincarePlot(axp2, pdata, mdata['allcorrdims'][:,0])


plt.tight_layout(h_pad=0.0, w_pad=0.0)
plt.tight_layout(h_pad=0.0, w_pad=0.0)


plt.savefig('edgeofchaos_plot.pdf', dpi=600)
plt.savefig('edgeofchaos_plot.png', dpi=600)