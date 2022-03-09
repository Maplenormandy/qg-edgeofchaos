# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:35:45 2021

@author: maple
"""

import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate
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

mpl.rc('font', **font)

mpl.rc('mathtext', fontset='cm')

# %% Figure generated via Scripts/Eigenfunctions/Trajectories/poincare_section_lyapunov.py

basedata = np.load('../poincare_input/poincare_config_fd_smooth.npz')
qbar = basedata['qbar']
uy = basedata['uy']

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

# %% Plot of contour superimposed on turbulence. First, load data

case = 1

with h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, 3-case), mode='r') as simdata:
    q = simdata['tasks/q'][0,:,:]
    x = simdata['scales/x']['1.0'][:]
    y = simdata['scales/y']['1.0'][:]
    xg, yg = np.meshgrid(x, y, indexing='ij')
    
offset = 1.475961757954208 # Magic number
if offset>np.pi:
    offset = offset-np.pi
qplot = q+8.0*(x[:,np.newaxis])

# Set up a new colorscale
colors = plt.cm.twilight(np.linspace(0,1,32))
numzones = 3
colors2 = np.vstack(list([colors for i in range(numzones+2)]))
qbrange = 8.0*(2*np.pi/numzones)*(numzones+2)
mymap2 = mpl.colors.LinearSegmentedColormap.from_list('twilight_stacked', colors2)



# %% Wave breaking section


tab20b = mpl.cm.get_cmap('tab20b')
tab20c = mpl.cm.get_cmap('tab20c')


fig = plt.figure(figsize=(8.7/2.54, 0.8*8.7/2.54), dpi=300)

gs_over = fig.add_gridspec(1, 2, width_ratios=[0.6, 0.4])

gs = gs_over[1].subgridspec(2, 1, hspace=0.6)


amps = ['040', '100', '120']
gsl = gs[0].subgridspec(len(amps), 1, hspace=0)


ax2 = fig.add_subplot(gs[1])

axshare = None

for ampind in range(len(amps)):
    data = np.load('../sections/case{}_breaking_amp{}.npz'.format(case, amps[ampind]))
    
    t = data['t']
    
    if axshare == None:
        ax = fig.add_subplot(gsl[ampind])
        axshare = ax
    else:
        ax = fig.add_subplot(gsl[ampind], sharey=axshare)
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
        
    ax.set_xlim(-np.pi, np.pi)
    #ax.set_ylim(-1.15, -0.45)
    #ax.set_ylim(ycenter-0.35, ycenter+0.35)
    ax.text(0.82, 0.7, '{0:.2f}'.format(float(amps[ampind])/100.0), transform=ax.transAxes)
    
    ax.xaxis.set_minor_locator(plt.NullLocator())
    if ampind == 0:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.set_title('PV Contour Evolution')
        ax.text(-0.2, 1.1, r'$y$', transform=ax.transAxes) 
    elif ampind == 1:
        ax.xaxis.set_major_locator(plt.NullLocator())
        #ax.set_ylabel('y')
    else:
        ax.set_xlabel(r'$x$')

    if 'y' in data:
        y = data['y']
        nparticles = y.shape[0]//2
        arclength = np.sum(np.sqrt(np.diff(y[:nparticles,:], axis=0)**2 + np.diff(y[nparticles:,:], axis=0)**2), axis=0)
        ax2.plot(range(len(arclength)), arclength/arclength[0], c=tab20c((2.5-ampind)/20.0), label='{0:.2f}'.format(float(amps[ampind])/100.0), marker='.')
    else:
        arclength = []
        for i in range(len(t)):
            yname = 'y{}'.format(i)
            if yname in data:
                yi = data['y{}'.format(i)]
                nparticles = len(yi)//2
                arclength.append(np.sum(np.sqrt(np.diff(yi[:nparticles])**2 + np.diff(yi[nparticles:])**2)))
            else:
                break
        ax2.plot(range(len(arclength)), np.array(arclength)/arclength[0], c=tab20c((2.5-ampind)/20.0), label='{0:.2f}'.format(float(amps[ampind])/100.0), marker='.')

#ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
#ax2.yaxis.set_minor_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
ax2.set_xlabel('Iteration')
ax2.text(-0.2, 1.0, r'$\ell/\ell_0$', transform=ax2.transAxes) 
ax2.set_title('Contour Length')
ax2.legend(loc='upper left')


plt.tight_layout(h_pad=0.6)
plt.tight_layout(h_pad=0.6)

#plt.savefig('wave_breaking.pdf', dpi=300)
#plt.savefig('wave_breaking.png', dpi=300)



#gs = gs_over[0].subgridspec(1, 1)
ax = fig.add_subplot(gs_over[0])

stride=4
#ax.pcolormesh(yg.T[::stride,::stride], xg.T[::stride,::stride], np.flipud(qplot.T)[::stride,::stride], cmap=mymap2, shading='gouraud', vmin=-qbrange/2.0-offset*8, vmax=qbrange/2.0-offset*8, rasterized=True)
ax.imshow(np.fliplr(qplot), origin='lower', cmap=mymap2, vmin=-qbrange/2.0-offset*8, vmax=qbrange/2.0-offset*8, extent=(-np.pi, np.pi, -np.pi, np.pi))
ax.set_aspect('equal')

for ampind in [1]:
    data = np.load('../sections/case{}_breaking_amp{}.npz'.format(case, amps[ampind]))
    
    if 'yclip' in data:
        #yclip = data['yclip']
        yclip0 = data['yclip'][:,0]
        yclipp = data['yclip'][:,plotind]
    else:
        yclip0 = data['yclip0']
        yclipp = data['yclip{}'.format(plotind)]
        
    t = data['t']
    nparticles0 = yclip0.shape[0]//2
    nparticlesp = yclipp.shape[0]//2
    
    stride = 1
    ax.plot(yclip0[nparticles0::stride], yclip0[:nparticles0:stride], c=tab20c(11.5/20.0), lw=1.0)
    
    
    
    
    chop = np.abs(np.diff(yclipp[nparticlesp::stride])) > 1.5*np.pi
    chopargs = np.argwhere(chop)[:,0] + 1
    
    c2 = tab20c(8.5/20.0)
    
    
    ax.plot(yclipp[nparticlesp:nparticlesp+chopargs[0]:stride], yclipp[:chopargs[0]:stride], c=c2, lw=1.0)
    ax.plot(yclipp[nparticlesp+chopargs[-1]::stride], yclipp[chopargs[-1]:nparticlesp:stride], c=c2, lw=1.0)
    
    for i in range(len(chopargs)-1):
        ax.plot(yclipp[nparticlesp+chopargs[i]:nparticlesp+chopargs[i+1]:stride], yclipp[chopargs[i]:chopargs[i+1]:stride], c=c2, lw=1.0)
    
ax.set_title('(a)', loc='left', size=6)
ax.text(0.05, 0.95, r'$q(x,y,t)$', transform=ax.transAxes, va='top')
ax.axis('off')
        
plt.tight_layout(h_pad=0.6)
plt.tight_layout(h_pad=0.6)

#plt.savefig('wave_overlay.pdf', dpi=300)
#plt.savefig('wave_overlay.png', dpi=300)
