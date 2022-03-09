# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:40:57 2021

@author: maple
"""

import concurrent.futures
#from multiprocessing import Pool

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from poincare_map import PoincareMapper

import scipy.interpolate
import scipy.signal

# %% Set up Baseline Poincare Mapper

#pm = PoincareMapper('poincare_input/poincare_config_validation.npz')
#numeigs = len(pm.data['kys'])

#pm.generateFullSection(np.ones(numeigs), np.zeros(numeigs), 'sections/section_validation.npz')



# %% Set up Poincare Mapper

suffix = '_uphavg'
#suffix = ''
case = 1
pm = PoincareMapper('poincare_input/case{}_poincare_config_fd_smooth{}.npz'.format(case, suffix))
numeigs = len(pm.data['kys'])

timedata = np.load('poincare_input/case{}_eigencomponent_timedata{}.npz'.format(case, suffix))

# %% Prepare Poincare section plots

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


# %% Poincare sections via amplitude of waves


def saveSection(ind):
    ampmult = timedata['ampdevs'][:,ind]
    phaseofs = timedata['phasedevs'][:,ind]
    
    np.savez('extra_poincare_sections/case{}_section_ind{:03d}{}.npz'.format(case, ind, suffix), ind)
    
    pm.generateFullSection(ampmult, phaseofs, 'extra_poincare_sections/case{}_section_ind{:03d}{}.npz'.format(case, ind, suffix), nparticles=15, sections=101, fancyspacing=True)
    data = np.load('extra_poincare_sections/case{}_section_ind{:03d}{}.npz'.format(case, ind, suffix))
    
    z0 = data['y'][:,0]
    yclip = data['yclip']
    
    nparticles = len(z0)//2
    colors = np.zeros((nparticles, yclip.shape[1]))
    
    rotation_number = (data['y'][nparticles:,-1] - data['y'][nparticles:,0]) / data['y'].shape[1] / 2 / np.pi
    xavg = np.average(data['y'][:nparticles,:], axis=1)
    
    rotcolors = np.zeros((yclip.shape[0]//2, yclip.shape[1]))
    
    # Compute index of shearless curves
    rotmins = np.zeros(uyminxs.shape, dtype=int)
    rotmaxs = np.zeros(uymaxxs.shape, dtype=int)
    
    for i in range(len(uyminxs)):
        rotmins[i] = np.argmin(rotation_number - (np.abs(xavg - uyminxs[i])<0.2)*1000.0)
        rotcolors[rotmins[i]:,:] += 0.5
        rotcolors[rotmins[i]+1:,:] += 0.5
    
    for i in range(len(uymaxxs)):
        rotmaxs[i] = np.argmax(rotation_number + (np.abs(xavg - uymaxxs[i])<0.2)*1000.0)
        rotcolors[rotmaxs[i]:,:] -= 0.5
        rotcolors[rotmaxs[i]+1:,:] -= 0.5
    
    # Compute "mixing lengths"
    stdresid = np.zeros(nparticles)


    for i in range(nparticles):
        xall = data['y'][i,:] - xavg[i]
        
        nvar = 9
        
        ymat = np.zeros((nvar, len(xall)-nvar))
        xmat = np.zeros((nvar, len(xall)-nvar))
        
        for j in range(nvar):
            if j == 0:
                ymat[j,:] = xall[nvar-j:]
            else:
                ymat[j,:] = xall[nvar-j:-j]
            
            xmat[j,:] = xall[nvar-j-1:-(j+1)]
        
        amat = ymat @ np.linalg.pinv(xmat)
        residuals = ymat - (amat @ xmat)
        
        stdresid[i] = np.sqrt(np.var(residuals[0,:]))
    
    if np.max(rotcolors) > 0.5:
        colors = stdresid[:, np.newaxis] * np.sign(rotcolors-0.5)
    else:
        colors = stdresid[:, np.newaxis] * np.sign(rotcolors+0.5)
    
    stride = 1
    stride2 = 1
    
    fig, ax = plt.subplots(1, 1, figsize=(10.0, 10.0))
    ax.set_aspect('equal', adjustable='datalim')
    
    ax.set_xlim([-np.pi,np.pi])
    ax.set_ylim([-np.pi,np.pi])
    
    plt.tight_layout()
    
    ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='Spectral', rasterized=True, vmin=-np.max(np.abs(colors)), vmax=np.max(np.abs(colors)))
    
    
    plt.savefig('extra_poincare_sections/case{}_section_ind{:03d}{}.png'.format(case, ind, suffix), dpi=100)
    
    return ind


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(saveSection, range(8))
        for result in results:
            print(result)


