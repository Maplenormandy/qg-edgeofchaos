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
    
    ax.set_aspect('equal', adjustable='datalim')
    
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])

# %% Plot figure

fig = plt.figure(figsize=(13.0/2.54*0.4, 13.0/2.54*0.36), dpi=300)
gs = fig.add_gridspec(1, 1)

case = 1

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

axq = fig.add_subplot(gs[0])
im = axq.imshow(np.fliplr(lenq(q+8*x[:,np.newaxis])), origin='lower', cmap='viridis', extent=(-np.pi, np.pi, -np.pi, np.pi))
divider = make_axes_locatable(axq)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)

cax.text(0.5, 1.03, '$\ell_q$', transform=cax.transAxes, ha='center', va='bottom')
#axq.set_yticklabels([])

#axq.set_title('DNS Snapshot')
#axq.text(0.0, 1.05, '(b)', transform=axq.transAxes, ha='left', va='bottom')

axq.set_xlabel('$x$')
axq.set_ylabel('$y$')


### Wave breaking ###
for suffix in ['qmin', 'qmax']:
    ldata = np.load('../sections/case{}_breaking_{}.npz'.format(case, suffix))
    breakingPlot(axq, ldata)

plt.tight_layout(pad=0)
plt.tight_layout(pad=0)

# %% 