# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:40:57 2021

@author: maple
"""

import numpy as np
from poincare_map import PoincareMapper
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl

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

# %% Set up Baseline Poincare Mapper

#pm = PoincareMapper('poincare_input/poincare_config_validation.npz')
#numeigs = len(pm.data['kys'])

#pm.generateFullSection(np.ones(numeigs), np.zeros(numeigs), 'sections/section_validation.npz')



# %% Set up Poincare Mapper

suffix = '_uphsort'
#suffix = ''
case = 2
pm = PoincareMapper('poincare_input/case{}_poincare_config_fd_smooth{}.npz'.format(case, suffix))
numeigs = len(pm.data['kys'])

# For plotting
qbar = pm.data['qbar']
uy = pm.data['uy']

nx = 2048
x = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)

# Set up interpolation functions
pad = 4
xp = np.zeros(nx+2*pad)
xp[pad:-pad] = x
xp[:pad] = x[-pad:] - 2*np.pi
xp[-pad:] = x[:pad] + 2*np.pi

def circularInterpolant(vec):
    vecp = np.zeros(nx+2*pad)
    vecp[pad:-pad] = vec
    vecp[:pad] = vec[-pad:]
    vecp[-pad:] = vec[:pad]
    
    return scipy.interpolate.interp1d(xp, vecp, kind='quadratic')

uyfft = np.fft.rfft(uy)
hilbuy = np.fft.irfft(1j*uyfft)
hilbuyf = circularInterpolant(hilbuy)
uyf = circularInterpolant(uy)

# Compute regions of zonal flow minima and maxima
uyminxs = x[scipy.signal.argrelextrema(uy, np.less)]
uymaxxs = x[scipy.signal.argrelextrema(uy, np.greater)]


def plotPoincare(section, filename):
    tab20b = mpl.cm.get_cmap('tab20b')
    tab20c = mpl.cm.get_cmap('tab20c')
    tab10 = mpl.cm.get_cmap('tab10')
    
    
    
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 10.0))
    #ax[1].scatter(uyf(x), x, c=np.mod(np.angle(uyf(x) + 1j*hilbuyf(x))*3,2*np.pi), cmap='twilight', marker='.')
    #ax[1].set_ylim([-np.pi, np.pi])
    
    data = np.load(section)
    
    z0 = data['y'][:,0]
    yclip = data['yclip']
    
    nparticles = len(z0)//2
    colors = np.zeros((nparticles, yclip.shape[1]))
    
    rotation_number = (data['y'][nparticles:,-1] - data['y'][nparticles:,0]) / data['y'].shape[1] / 2 / np.pi
    xavg = np.average(data['y'][:nparticles,:], axis=1)
    xstd = np.sqrt(np.var(data['y'][:nparticles,:], axis=1))
    
    stride = 1
    stride2 = 1
    colors[:,:] = np.mod(np.angle(uyf(z0[:nparticles]) + 1j*hilbuyf(z0[:nparticles]))*3,2*np.pi)[:,np.newaxis]
    #colors[:,:] = (np.mod(np.arange(nparticles), 10) / 10.0 + 0.05)[:,np.newaxis]
    #colors[:,:] = np.sign(np.roll(rotation_number,1) - np.roll(rotation_number,-1))[:, np.newaxis]
    #colors = np.zeros((yclip.shape[0]//2, yclip.shape[1]))
    
    # Compute index of shearless curves
    rotmins = np.zeros(uyminxs.shape, dtype=int)
    rotmaxs = np.zeros(uymaxxs.shape, dtype=int)
    
    for i in range(len(uyminxs)):
        rotmins[i] = np.argmin(rotation_number - (np.abs(xavg - uyminxs[i])<0.2)*1000.0)
        colors[rotmins[i]:,:] += 0.5
        colors[rotmins[i]+1:,:] += 0.5
    
    for i in range(len(uymaxxs)):
        rotmaxs[i] = np.argmax(rotation_number + (np.abs(xavg - uymaxxs[i])<0.2)*1000.0)
        colors[rotmaxs[i]:,:] -= 0.5
        colors[rotmaxs[i]+1:,:] -= 0.5
    
    ax.set_aspect('equal', adjustable='datalim')
    ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='twilight', rasterized=True)
    #ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='brg', rasterized=True)
    ax.set_xlim([-np.pi,np.pi])
    ax.set_ylim([-np.pi,np.pi])
    
    plt.tight_layout()
    
    plt.savefig(filename, dpi=100)
    
    plt.close()


# %% Poincare sections

amprange = ['100', '110']

numwaves = numeigs

for i in range(len(amprange)):
    m = float(amprange[i])/100.0
    print('sections/case{}_section_amp{}{}.npz'.format(case,amprange[i], suffix))
    
    ampmult = np.ones(numeigs)*m
    ampmult[numwaves:] = 0
    
    sectionfile = 'sections/case{}_section_amp{}{}.npz'.format(case,amprange[i], suffix)
    plotfile = 'extra_poincare_sections//case{}_section_amp{}{}.png'.format(case,amprange[i], suffix)
    
    #pm.generateFullSection(np.ones(numeigs)*m, np.zeros(numeigs), 'sections/case{}_section_amp{}{}.npz'.format(case,amprange[i], suffix), nparticles=193, sections=3109, fancyspacing=True)
    pm.generateFullSection(ampmult, np.zeros(numeigs), sectionfile, nparticles=521, sections=521, fancyspacing=True)

    plotPoincare(sectionfile, plotfile)