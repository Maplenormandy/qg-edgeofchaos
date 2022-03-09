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
        'size'   : 18}

mpl.rc('font', **font)

# %% 

case = 1
basedata = np.load('../poincare_input/case{}_poincare_config_fd_smooth.npz'.format(case))
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



# %% Zonal flows

"""
fig, ax = plt.subplots(1, 1, figsize=(3.0, 10.0))
ax.scatter(uyf(x), x, c=np.mod(np.angle(uyf(x) + 1j*hilbuyf(x))*3,2*np.pi), cmap='twilight', marker='.')
ax.set_ylim([-np.pi, np.pi])

plt.tight_layout()

#plt.savefig('poincare_section_zonalflow.pdf', dpi=100)
#plt.savefig('poincare_section_zonalflow.png', dpi=100)
"""

# %% Poincare Section

tab20b = mpl.cm.get_cmap('tab20b')
tab20c = mpl.cm.get_cmap('tab20c')



fig, ax = plt.subplots(1, 1, figsize=(10.0, 10.0))
#ax[1].scatter(uyf(x), x, c=np.mod(np.angle(uyf(x) + 1j*hilbuyf(x))*3,2*np.pi), cmap='twilight', marker='.')
#ax[1].set_ylim([-np.pi, np.pi])

suffix = 'amp000'
data = np.load('../sections/case{}_section_{}.npz'.format(case,suffix))

z0 = data['y'][:,0]
yclip = data['yclip']

nparticles = len(z0)//2
colors = np.zeros((nparticles, yclip.shape[1]))

stride = 1
stride2 = 1
colors[:,:] = np.mod(np.angle(uyf(z0[:nparticles]) + 1j*hilbuyf(z0[:nparticles]))*3,2*np.pi)[:,np.newaxis]

ax.set_aspect('equal', adjustable='datalim')
ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='twilight', rasterized=True)
ax.set_xlim([-np.pi,np.pi])
ax.set_ylim([-np.pi,np.pi])

plt.tight_layout()
#plt.tight_layout(h_pad=0.6)

plt.savefig('poincare_section_case{}_{}.pdf'.format(case, suffix), dpi=100)
plt.savefig('poincare_section_case{}_{}.png'.format(case, suffix), dpi=100)