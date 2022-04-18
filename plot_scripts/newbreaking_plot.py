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
    

def breakingPlot(ax, data):
    tab20c = mpl.cm.get_cmap('tab20c')
    
    
    t = data['t']
    stride = 1
    plotind = 1
    
    if 'yclip' in data:
        #yclip = data['yclip']
        yclip0 = data['yclip'][:,0]
        yclipp = data['yclip'][:,plotind]
    else:
        yclip0 = data['yclip0']
        yclipp = data['yclip{}'.format(plotind)]
        
    nparticles0 = yclip0.shape[0]//2
    nparticlesp = yclipp.shape[0]//2
    
    ax.plot(yclip0[nparticles0::stride], yclip0[:nparticles0:stride], c=tab20c(11.5/20.0), lw=1.0)
    
    ycenter = (np.max(yclip0[:nparticles0:stride]) - np.min(yclip0[:nparticles0:stride]))

    chop = np.abs(np.diff(yclipp[nparticlesp::stride])) > 1.5*np.pi
    chopargs = np.argwhere(chop)[:,0] + 1
    
    c2 = tab20c(8.5/20.0)
    
    ax.plot(yclipp[nparticlesp:nparticlesp+chopargs[0]:stride], yclipp[:chopargs[0]:stride], c=c2, lw=1.0)
    ax.plot(yclipp[nparticlesp+chopargs[-1]::stride], yclipp[chopargs[-1]:nparticlesp:stride], c=c2, lw=1.0)
    
    for i in range(len(chopargs)-1):
        ax.plot(yclipp[nparticlesp+chopargs[i]:nparticlesp+chopargs[i+1]:stride], yclipp[chopargs[i]:chopargs[i+1]:stride], c=c2, lw=1.0)
        
    
    ax.set_aspect('equal', adjustable='datalim')
    
    ax.set_xlim([-np.pi, np.pi])
    ax.set_ylim([-np.pi, np.pi])

# %% Plot figure

fig = plt.figure(figsize=(17.8/2.54, 0.25*17.8/2.54), dpi=300)
gs = fig.add_gridspec(1, 4)

case = 1

cdata = np.load('case{}_snapcontours.npz'.format(case))
mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))
qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))

### Load the PV data ###
snapind = 51
snapfilenum = (snapind//16+2)
simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, snapfilenum), 'r')
qindex = (snapind%16)

q = simdata['tasks/q'][qindex,:,:]

lenq = circularInterpolant(cdata['levels'], np.average(cdata['lenmaxcontour'], axis=0), 2*8*np.pi, 500)


### Plot the poincare sections ###
pdata = np.load('../extra_poincare_sections/case{}_section_ind{:03d}_uphavg.npz'.format(case,snapind))
axp1 = fig.add_subplot(gs[0])

poincarePlot(axp1, pdata, mdata['allcorrdims'][:,snapind])


### Plot of the q contours ###
axq = fig.add_subplot(gs[1])
im = axq.imshow(np.fliplr(lenq(q+8*x[:,np.newaxis])), origin='lower', cmap='viridis', extent=(-np.pi, np.pi, -np.pi, np.pi))
divider = make_axes_locatable(axq)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)

cax.text(0.5, 1.05, '$\ell_q$', transform=cax.transAxes, ha='center', va='bottom')
axq.set_yticklabels([])


### Wave breaking ###
axw = fig.add_subplot(gs[2])

for suffix in ['qmin', 'qmax']:
    ldata = np.load('../sections/case{}_breaking_{}.npz'.format(case, suffix))
    breakingPlot(axw, ldata)


"""
### Chaotic fraction vs. space ###
gsinner = gs[1,1].subgridspec(2,1)

axm = fig.add_subplot(gsinner[0])

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

### Chaotic fraction vs space-time ###
axf = fig.add_subplot(gsinner[1])

xsort = np.argsort(mdata['allxavgs'], axis=0)
xplot = np.take_along_axis(mdata['allxavgs'], xsort, axis=0)
corrplot = np.take_along_axis(mdata['allcorrdims'], xsort, axis=0)
trange = np.arange(0, 64.1, 0.25)
trange2 = np.arange(-0.125, 64.0 + 0.125 + 0.01, 0.25)
   
for i in range(len(trange)):
    xplot2 = recenter(xplot[:,i])
    tplot2 = np.array([trange2[i], trange2[i+1]])
    
    tplot3, xplot3 = np.meshgrid(tplot2, xplot2)
    
    axf.pcolormesh(tplot3, xplot3, np.array([corrplot[:,i]]).T, vmin=1.0, vmax=2.0, shading='flat')

if (case==1):
    axf.set_ylim([1.0, 3.0])
else:
    axf.set_ylim([-1.3, -0.5])
    
plt.tight_layout(h_pad=0.0, w_pad=0.0)
plt.tight_layout(h_pad=0.0, w_pad=0.0)

plt.savefig('newbreaking_plot.pdf', dpi=900)
plt.savefig('newbreaking_plot.png', dpi=900)
"""