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

import os
os.chdir('../')

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

def poincarePlot(ax, pdata, colordata, labelcax=True):
    z0 = pdata['y'][:,0]
    yclip = pdata['yclip']
    
    nparticles = len(z0)//2
    colors = np.zeros((nparticles, yclip.shape[1]))
    colors[:,:] = colordata[:, np.newaxis]
    ax.set_aspect('equal', adjustable='datalim')

    #ax.set_xticks([])
    sc = ax.scatter(yclip[nparticles:,:], yclip[:nparticles,:], s=(72.0/450.0)**2, marker='o', linewidths=0, c=colors[:,:], rasterized=True, cmap='viridis', vmin=1.0, vmax=2.0)
    
    ax.set_xlim([-np.pi,np.pi])
    ax.set_ylim([-np.pi,np.pi])
    
    


# %% Plot figure

fig = plt.figure(figsize=(6.0/2.54, 5.0/2.54), dpi=300)


#fig = plt.figure(figsize=(17.8/2.54*(2.0/3.0), 0.32*17.8/2.54), dpi=300)
#gs = fig.add_gridspec(1, 2, width_ratios=[2,2])

case = 2

cdata = np.load('case{}_snapcontours.npz'.format(case))
mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))
qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))

### Load PV data ###
#snapind = 51 if (case == 1) else 192
snapind = 0 if (case == 1) else 192
#snapind = 0
snapfilenum = (snapind//16+2) if (case == 1) else (snapind//10+1)
simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, snapfilenum), 'r')
qindex = (snapind%16) if (case == 1) else (snapind%10)

q = simdata['tasks/q'][qindex,:,:]

lenq = circularInterpolant(cdata['levels'], np.average(cdata['lenmaxcontour'], axis=0), 2*8*np.pi, 500)


### Plot the poincare plots max amplitude ###
pdata = np.load('../extra_poincare_sections/case{}_section_ind{:03d}_uphavg.npz'.format(case,snapind))
axp1 = fig.add_subplot(1, 1, 1)

poincarePlot(axp1, pdata, mdata['allcorrdims'][:,snapind])





### Plot the DNS snapshot ###
axq = axp1

qplot = np.fliplr(lenq(q+8*x[:,np.newaxis]))
qplotcolor = mpl.cm.viridis((qplot-np.min(qplot))/(np.max(qplot)-np.min(qplot)))

x = np.linspace(-np.pi, np.pi, num=2048, endpoint=False)
trans = (1.0-np.tanh(2*x))/2.0


qplotcolor[:,:,3] = trans[np.newaxis,:]
#im = axq.imshow(qplot, origin='lower', cmap='viridis', extent=(-np.pi, np.pi, -np.pi, np.pi))
im = axq.imshow(qplotcolor[:,:,:], origin='lower', extent=(-np.pi, np.pi, -np.pi, np.pi), zorder=2.5)
divider = make_axes_locatable(axq)




### Plot the boundaries of the chaotic region ###
if case == 1:
    pcplot = [132, 186]
else:
    pcplot = [61, 80, 143, 166] # snapind = 192
    #pcplot = [60, 81, 146, 166]
#pcplot = [141, 172]
yclip = pdata['yclip']

for ind in pcplot:
    nparticles = yclip.shape[0]//2
    xind = np.argsort(yclip[nparticles+ind,:])
    xplot = (yclip[nparticles+ind,:])[xind]
    yplot = (yclip[ind,:])[xind]
    axp1.plot(xplot[::4], yplot[::4], c='tab:orange', ls='--', lw=0.9, zorder=3)
    #axq.plot(xplot[::4], yplot[::4], c='tab:orange', ls='--', lw=0.9)
    pass

axp1.set_xlim([-np.pi,np.pi])
axp1.set_ylim([-4*np.pi/6,np.pi])

axp1.set_xticks([])
axp1.set_yticks([])
axp1.spines['top'].set_visible(False)
axp1.spines['right'].set_visible(False)
axp1.spines['bottom'].set_visible(False)
axp1.spines['left'].set_visible(False)

plt.tight_layout(pad=0)

plt.savefig('graphical_abstract.pdf', dpi=300)
#plt.savefig('mixing_plot.png', dpi=300)