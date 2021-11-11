# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:35:45 2021

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate

font = {'family' : 'serif',
        'size'   : 6}

mpl.rc('font', **font)


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




# %% Wave breaking section

tab20b = mpl.cm.get_cmap('tab20b')
tab20c = mpl.cm.get_cmap('tab20c')

fig = plt.figure(figsize=(3.375, 3.375*0.5), dpi=300)
gs = fig.add_gridspec(1, 2)


amps = ['040', '100']
gsl = gs[0].subgridspec(len(amps), 1, hspace=0)


ax2 = fig.add_subplot(gs[1])

for ampind in range(len(amps)):
    data = np.load('../sections/breaking_amp{}.npz'.format(amps[ampind]))
    yclip = data['yclip']
    y = data['y']
    t = data['t']
    nparticles = yclip.shape[0]//2
    
    ax = fig.add_subplot(gsl[ampind])
    stride = 1
    ax.plot(yclip[nparticles::stride,0], yclip[:nparticles:stride,0], c=tab20b(3.5/20.0), lw=1.0)
    
    plotind = -1
    
    
    chop = np.abs(np.diff(yclip[nparticles::stride,plotind])) > 1.5*np.pi
    chopargs = np.argwhere(chop)[:,0] + 1
    
    c2 = tab20b(0.5/20.0)
    
    ax.plot(yclip[nparticles:nparticles+chopargs[0]:stride,plotind], yclip[:chopargs[0]:stride,plotind], c=c2, lw=1.0)
    ax.plot(yclip[nparticles+chopargs[-1]::stride,plotind], yclip[chopargs[-1]:nparticles:stride,plotind], c=c2, lw=1.0)
    
    for i in range(len(chopargs)-1):
        ax.plot(yclip[nparticles+chopargs[i]:nparticles+chopargs[i+1]:stride,plotind], yclip[chopargs[i]:chopargs[i+1]:stride,plotind], c=c2, lw=1.0)
        
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.09, -0.59)
    ax.text(2.0, -0.7, '{0:.2f}'.format(float(amps[ampind])/100.0))
    
    ax.xaxis.set_minor_locator(plt.NullLocator())
    if ampind == 0:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.set_title('PV Contour Evolution')
    elif ampind == 1:
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.set_ylabel('y')
    else:
        ax.set_xlabel('x')

    arclength = np.sum(np.sqrt(np.diff(y[:nparticles,:], axis=0)**2 + np.diff(y[nparticles:,:], axis=0)**2), axis=0)
    ax2.semilogy(range(len(arclength)), arclength, c=tab20c((2.5-ampind)/20.0), label='{0:.2f}'.format(float(amps[ampind])/100.0), marker='.')

ax2.set_xlabel('Iteration')
#ax2.text(-0.1, 1.02, r'Contour Length', transform=ax2.transAxes) 
ax2.set_title('Contour Length')
ax2.legend(loc='upper left')


plt.tight_layout(h_pad=0.6)
plt.tight_layout(h_pad=0.6)

plt.savefig('wave_breaking.pdf', dpi=200)
plt.savefig('wave_breaking.png', dpi=300)