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
import scipy.spatial
import scipy.stats


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


from sklearn import manifold

# %% 

case = 2
basedata = np.load('../poincare_input/case{}_poincare_config_fd_smooth_uphavg.npz'.format(case))
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

suffix = 'ind050_uphavg'
data = np.load('../extra_poincare_sections/case{}_section_{}.npz'.format(case,suffix))

z0 = data['y'][:,0]
yclip = data['yclip']
yorig = data['y']

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

rotmap = np.zeros(nparticles)

# Compute index of shearless curves
rotmins = np.zeros(uyminxs.shape, dtype=int)
rotmaxs = np.zeros(uymaxxs.shape, dtype=int)


for i in range(len(uyminxs)):
    rotmins[i] = np.argmin(rotation_number - (np.abs(xavg - uyminxs[i])<0.2)*1000.0)
    
    rotmap += (xavg >= xavg[rotmins[i]]) * 0.5
    rotmap += (xavg > xavg[rotmins[i]]) * 0.5

for i in range(len(uymaxxs)):
    rotmaxs[i] = np.argmax(rotation_number + (np.abs(xavg - uymaxxs[i])<0.2)*1000.0)
    
    rotmap -= (xavg >= xavg[rotmaxs[i]]) * 0.5
    rotmap -= (xavg > xavg[rotmaxs[i]]) * 0.5

rotcolors[:,:] = rotmap[:,np.newaxis]

colors=rotcolors

"""
fig, ax = plt.subplots(1, 1, figsize=(10.0, 10.0))
ax.set_aspect('equal', adjustable='datalim')

ax.set_xlim([-np.pi,np.pi])
ax.set_ylim([-np.pi,np.pi])

plt.tight_layout()

#ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='twilight', rasterized=True)
ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='brg', rasterized=True)
"""

#plt.tight_layout(h_pad=0.6)


#plt.savefig('poincare_scratch/case{}_section_{}.png'.format(case,suffix), dpi=100)

# %%


fig, ax = plt.subplots(1, 1, figsize=(3.0, 10.0))
ax.scatter(uyf(x), x, c=np.mod(np.angle(uyf(x) + 1j*hilbuyf(x))*3,2*np.pi), cmap='twilight', marker='.')
ax.set_ylim([-np.pi, np.pi])


plt.scatter(-rotation_number*basedata['freq'] + basedata['dopplerc'], xavg, marker='.', c=colors[:,0], cmap='brg')
plt.tight_layout()


#plt.savefig('poincare_section_zonalflow.pdf', dpi=100)
#plt.savefig('poincare_section_zonalflow.png', dpi=100)

# %%

"""
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
ax.set_xlim([-np.pi,np.pi])
ax.set_ylim([-np.pi,np.pi])

plt.tight_layout()

#ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='twilight', rasterized=True)
ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='Spectral', rasterized=True, vmin=-np.max(np.abs(colors)), vmax=np.max(np.abs(colors)))
"""

# %%

#fig, ax = plt.subplots(1, 1, figsize=(3.0, 10.0))
#ax.scatter(xstd, xavg)
#ax.scatter(stdresid, xavg, c=colors[:,0], cmap='Spectral', vmin=-np.max(np.abs(colors)), vmax=np.max(np.abs(colors)))
#ax.scatter(stdresid, xavg, c=rotcolors[:,0], cmap='brg')
#ax.set_ylim([-np.pi, np.pi])

# %% 3d plots of data

# 46 for a chaotic orbit, 


#plt.figure()
#plt.scatter(yclip[pind+nparticles,:], yclip[pind,:], s=72.0/fig.dpi, marker='o')




pind = 41
samples = np.array([np.cos(yorig[pind+nparticles,:]), np.sin(yorig[pind+nparticles,:]), yorig[pind,:]])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(samples[0,:], samples[1,:], samples[2,:])

#embedding = manifold.LocallyLinearEmbedding(method="hessian", n_neighbors=10, n_components=2)
#X_transformed = embedding.fit_transform(samples.T)

#plt.figure()
#plt.scatter(X_transformed[:,0], X_transformed[:,1])

sx = np.mod(yorig[pind+nparticles,:], 2*np.pi)
sy = yorig[pind,:]

sxd = np.mod(scipy.spatial.distance.pdist(np.array([sx]).T)+np.pi, 2*np.pi)-np.pi
syd = scipy.spatial.distance.pdist(np.array([sy]).T)
pdists = np.sqrt(sxd**2 + syd**2)

psorted = np.sort(pdists)
ncorr = np.arange(1, len(psorted)+1)

plt.figure()
plt.loglog(psorted[:], ncorr[:], marker='.')

bounds = list(map(lambda x: int(np.round(x)), np.geomspace(16, len(psorted+1), 32)))

for b in bounds:
    plt.axhline(b)



# %%

def fit_slope(lind, rind, psorted, bounds):
    lbound = bounds[lind]
    ubound = bounds[rind]
    
    sampinds = np.array(list(map(lambda x: int(np.round(x)), np.geomspace(lbound, ubound, num=256))), dtype=int)
    result = scipy.stats.linregress(np.log(psorted[sampinds-1]), np.log(ncorr[sampinds-1]))
    
    return result




#result = scipy.stats.linregress(np.log(psorted[sampinds]), np.log(ncorr[sampinds]))
#print(result.slope)

#plt.loglog(psorted[:], np.exp(np.log(psorted[:])*result.slope + result.intercept))
#plt.axhline(lbound)
#plt.axhline(ubound)

#plt.figure()
#plt.hist(np.log((np.gradient(np.log(ncorr)) / np.gradient(np.log(psorted)))), bins=128)


# %% Correlation dimension scatter plot

corrdim = np.zeros(nparticles)

print("Correlation dimension")
for i in range(nparticles):
    if i%10 == 0:
        print(i)
        
    sx = np.mod(yorig[i+nparticles,:], 2*np.pi)
    sy = yorig[i,:]
    
    sxd = np.mod(scipy.spatial.distance.pdist(np.array([sx]).T)+np.pi, 2*np.pi)-np.pi
    syd = scipy.spatial.distance.pdist(np.array([sy]).T)
    pdists = np.sqrt(sxd**2 + syd**2)
    psorted = np.sort(pdists)
    ncorr = np.arange(1, len(psorted)+1)
    
    bounds = list(map(lambda x: int(np.round(x)), np.geomspace(16, len(psorted+1), 32)))
    rsq = []
    slope = []
    
    lind = 0
    rind = len(bounds)-1
    
    
    result = fit_slope(lind, rind, psorted, bounds)

    rsq.append(result.rvalue**2)
    slope.append(result.slope)
    
    while rsq[-1] < 0.999 and (rind-lind)>16:
        resultl = fit_slope(lind+1, rind, psorted, bounds)
        resultr = fit_slope(lind, rind-1, psorted, bounds)
        
        if resultl.rvalue**2 > resultr.rvalue**2:
            lind = lind+1
            result = resultl
        else:
            rind = rind-1
            result = resultr
    
        rsq.append(result.rvalue**2)
        slope.append(result.slope)
    
    corrdim[i] = slope[-1]


# %% New correlation dimension

"""
i = pind
        
sx = np.mod(yorig[i+nparticles,:], 2*np.pi)
sy = yorig[i,:]

sxd = np.mod(scipy.spatial.distance.pdist(np.array([sx]).T)+np.pi, 2*np.pi)-np.pi
syd = scipy.spatial.distance.pdist(np.array([sy]).T)
pdists = np.sqrt(sxd**2 + syd**2)
psorted = np.sort(pdists)
ncorr = np.arange(1, len(psorted)+1)

#plt.loglog(psorted[:], ncorr[:], marker='.')

tr = 1.0 / ((np.cumsum(np.log(psorted)) - ncorr*np.log(psorted))[1:] * -1 / (ncorr-1)[1:])

plt.figure()
plt.semilogx(psorted[1:], tr)
"""


# %%

fig, ax = plt.subplots(1, 1, figsize=(10.0, 10.0))
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlim([-np.pi,np.pi])
ax.set_ylim([-np.pi,np.pi])

plt.tight_layout()

if np.max(rotcolors) > 0.5:
    colors = (corrdim[:,np.newaxis] > 1.4) * np.sign(rotcolors-0.5)
else:
    colors = (corrdim[:,np.newaxis] > 1.4) * np.sign(rotcolors+0.5)

colors[:,:] = corrdim[:,np.newaxis]
#ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='twilight', rasterized=True)
sc = ax.scatter(yclip[nparticles::stride,::stride2], yclip[:nparticles:stride,::stride2], s=72.0/fig.dpi, marker='o', linewidths=0, c=colors[::stride,::stride2], cmap='viridis', rasterized=True)
#plt.colorbar(sc)


# %%
fig, ax = plt.subplots(1, 1, figsize=(10.0, 3.0))
#ax.scatter(range(nparticles), stdresid, c=rotcolors[:,0], cmap='brg')
plt.plot(corrdim, marker='.', ls='')
plt.tight_layout()