# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:35:45 2021

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate
import scipy.signal


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

# %% 

case = 2
basedata = np.load('../poincare_input/case{}_poincare_config_fd_smooth_coherent.npz'.format(case))
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

# Compute regions of zonal flow minima and maxima
uyminxs = x[scipy.signal.argrelextrema(uy, np.less)]
uymaxxs = x[scipy.signal.argrelextrema(uy, np.greater)]


# %% Poincare Section

tab20b = mpl.cm.get_cmap('tab20b')
tab20c = mpl.cm.get_cmap('tab20c')
tab10 = mpl.cm.get_cmap('tab10')



#ax[1].scatter(uyf(x), x, c=np.mod(np.angle(uyf(x) + 1j*hilbuyf(x))*3,2*np.pi), cmap='twilight', marker='.')
#ax[1].set_ylim([-np.pi, np.pi])

suffix = 'amp100_energymin'
data = np.load('../sections/case{}_section_{}.npz'.format(case,suffix))

z0 = data['y'][:,0]
yclip = data['yclip']

nparticles = len(z0)//2
colors = np.zeros((nparticles, yclip.shape[1]))

rotation_number = (data['y'][nparticles:,-1] - data['y'][nparticles:,0]) / data['y'].shape[1] / 2 / np.pi
xavg = np.average(data['y'][:nparticles,:], axis=1)
xstd = np.sqrt(np.var(data['y'][:nparticles,:], axis=1))

stride = 1
stride2 = 1
#colors[:,:] = np.mod(np.angle(uyf(z0[:nparticles]) + 1j*hilbuyf(z0[:nparticles]))*3,2*np.pi)[:,np.newaxis]
#colors[:,:] = (np.mod(np.arange(nparticles), 10) / 10.0 + 0.05)[:,np.newaxis]
#colors[:,:] = np.sign(np.roll(rotation_number,1) - np.roll(rotation_number,-1))[:, np.newaxis]
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


colors=rotcolors

fig, ax = plt.subplots(1, 1, figsize=(10.0, 10.0))
ax.set_aspect('equal', adjustable='datalim')
#ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='twilight', rasterized=True)
ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='brg', rasterized=True)
ax.set_xlim([-np.pi,np.pi])
ax.set_ylim([-np.pi,np.pi])

plt.tight_layout()
#plt.tight_layout(h_pad=0.6)


plt.savefig('poincare_scratch/case{}_section_{}.png'.format(case,suffix), dpi=100)

# %%


fig, ax = plt.subplots(1, 1, figsize=(3.0, 10.0))
ax.scatter(uyf(x), x, c=np.mod(np.angle(uyf(x) + 1j*hilbuyf(x))*3,2*np.pi), cmap='twilight', marker='.')
ax.set_ylim([-np.pi, np.pi])


plt.scatter(-rotation_number*basedata['freq'] + basedata['dopplerc'], xavg, marker='.', c=colors[:,0], cmap='brg')
plt.tight_layout()


#plt.savefig('poincare_section_zonalflow.pdf', dpi=100)
#plt.savefig('poincare_section_zonalflow.png', dpi=100)

# %%


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
    
    #plt.plot(xall[nvar:])
    #plt.plot(residuals[0,:])



# %%

if np.max(rotcolors) > 0.5:
    colors = stdresid[:, np.newaxis] * np.sign(rotcolors-0.5)
else:
    colors = stdresid[:, np.newaxis] * np.sign(rotcolors+0.5)
fig, ax = plt.subplots(1, 1, figsize=(10.0, 10.0))
ax.set_aspect('equal', adjustable='datalim')
#ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='twilight', rasterized=True)
ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='Spectral', rasterized=True, vmin=-np.max(np.abs(colors)), vmax=np.max(np.abs(colors)))
ax.set_xlim([-np.pi,np.pi])
ax.set_ylim([-np.pi,np.pi])

plt.tight_layout()

# %%

fig, ax = plt.subplots(1, 1, figsize=(3.0, 10.0))
#ax.scatter(xstd, xavg)
ax.scatter(stdresid, xavg, c=colors[:,0], cmap='Spectral')
ax.set_ylim([-np.pi, np.pi])
plt.tight_layout()
# %%
"""
plt.figure()
plotind = np.argmax(np.max(data['y'][:nparticles,:], axis=1)-np.min(data['y'][:nparticles,:], axis=1))
plt.scatter(yclip[plotind+nparticles,:], yclip[plotind,:], s=72.0/fig.dpi, marker='o')
plt.axis('equal')
"""