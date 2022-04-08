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
    #ax.set_yticks([])
    sc = ax.scatter(yclip[nparticles:,:], yclip[:nparticles,:], s=(72.0/600.0)**2, marker='o', linewidths=0, c=colors[:,:], rasterized=True, cmap='viridis', vmin=1.0, vmax=2.0)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(sc, cax=cax)
    
    ax.set_xlim([-np.pi,np.pi])
    ax.set_ylim([-np.pi,np.pi])
    

# %% Set up the figure

fig = plt.figure(figsize=(17.8/2.54, 0.25*17.8/2.54), dpi=300)
gs = fig.add_gridspec(1, 4, width_ratios=[2,2,2,2])


# %% Plot figure

for case in [1,2]:
    cdata = np.load('case{}_snapcontours.npz'.format(case))
    mdata = np.load('case{}_mixing_lengths.npz'.format(case))
    qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))

    uy = np.fft.irfft(np.fft.rfft(qbars['qequiv']-8*x[np.newaxis,:], axis=1)*kxinv, axis=1)
    
    uymininds = scipy.signal.argrelmin(uy, axis=1, mode='wrap')
    uymaxinds = scipy.signal.argrelmax(uy, axis=1, mode='wrap')
    
    numbands = 2 if (case == 1) else 3

    ### Determine chaotic fraction of Poincare section orbits ###
    chaoticfraction = np.zeros((numbands, uy.shape[0]))
    
    for i in range(numbands):
        uymin = uy[uymininds[0][i::numbands], uymininds[1][i::numbands]]
        xmin = x[uymininds[1][i::numbands]]
        #print(xmin)
        
        uymax0 = uy[uymaxinds[0][i::numbands], uymaxinds[1][i::numbands]]
        xmax0 = x[uymaxinds[1][i::numbands]]
        if uymaxinds[1][i] > uymininds[1][i]:
            ind2 = (i-1)%numbands
        else:
            ind2 = (i+1)%numbands
        
        uymax1 = uy[uymaxinds[0][ind2::numbands], uymaxinds[1][ind2::numbands]]
        xmax1 = x[uymaxinds[1][ind2::numbands]]
        
        uymax = (uymax0 + uymax1) / 2.0
        
        xmax2 = np.max(np.array([xmax0, xmax1]), axis=0)
        xmax3 = np.min(np.array([xmax0, xmax1]), axis=0)
        
        #print(xmax3[0], xmax2[0], xmin[0])
        
        if xmin[0] < xmax3[0] or xmin[0] > xmax2[0]:
            cutoff = (mdata['allxavgs'] <= xmax3[np.newaxis,:]) + (mdata['allxavgs'] >= xmax2[np.newaxis,:])
        else:
            cutoff = (mdata['allxavgs'] >= xmax3[np.newaxis,:]) * (mdata['allxavgs'] <= xmax2[np.newaxis,:])
            
        chaoticfraction[i, :] = np.sum((mdata['allcorrdims']>1.5)*cutoff, axis=0) / np.sum(cutoff, axis=0)



    ### Load and plot PV data ###
    snapfilenum = 5 if (case == 1) else 6
    simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, snapfilenum), 'r')
    qindex = (51%16) if (case == 1) else (50%10)
    
    q = simdata['tasks/q'][qindex,:,:]
    
    lenq = circularInterpolant(cdata['levels'], np.average(cdata['lenmaxcontour'], axis=0), 2*8*np.pi, 500)
    
    axq = fig.add_subplot(gs[2*case-2])
    #axq.set_xticks([])
    #axq.set_yticks([])
        
    im = axq.imshow(np.fliplr(lenq(q+8*x[:,np.newaxis])), origin='lower', cmap='viridis', extent=(-np.pi, np.pi, -np.pi, np.pi))
    divider = make_axes_locatable(axq)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(im, cax=cax)
    
    
    ### Plot the poincare plots max amplitude ###
    pind = '051' if (case==1) else '050'
    pdata = np.load('../extra_poincare_sections/case{}_section_ind{}_uphavg.npz'.format(case,pind))
    axp1 = fig.add_subplot(gs[2*case-1])
    
    poincarePlot(axp1, pdata, mdata['allcorrdims'][:,int(pind)])
    
plt.tight_layout(h_pad=0.0, w_pad=0.0)
plt.tight_layout(h_pad=0.0, w_pad=0.0)

plt.savefig('mixing_plot.pdf', dpi=1000)
plt.savefig('mixing_plot.png', dpi=1000)