# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:33:46 2022

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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

case = 1
cdata = np.load('case{}_snapcontours.npz'.format(case))
data = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))
qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))

stdresids = data['allstdresids']
ranresids = data['allranresids']

nx = 2048
x = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)
kx = np.fft.rfftfreq(nx, 1.0/nx)
kxinv = np.zeros(kx.shape, dtype=complex)
kxinv[kx>0] = 1.0/(-1j*kx[kx>0])

uy = np.fft.irfft(np.fft.rfft(qbars['qequiv']-8*x[np.newaxis,:], axis=1)*kxinv, axis=1)

uymininds = scipy.signal.argrelmin(uy, axis=1, mode='wrap')
uymaxinds = scipy.signal.argrelmax(uy, axis=1, mode='wrap')

if case == 1:
    numbands = 2
else:
    numbands = 3


zonalamps = np.zeros((numbands, uy.shape[0]))
mixinglengths = np.zeros((numbands, uy.shape[0]))
zonallengths = np.zeros((numbands, uy.shape[0]))
chaoticfraction = np.zeros((numbands, uy.shape[0]))
    
for i in range(numbands):
    uymin = uy[uymininds[0][i::numbands], uymininds[1][i::numbands]]
    xmin = x[uymininds[1][i::numbands]]
    #print(xmin)
    
    uymax0 = uy[uymaxinds[0][i::numbands], uymaxinds[1][i::numbands]]
    xmax0 = x[uymaxinds[1][i::numbands]]
    if uymaxinds[1][i] > uymininds[1][i]:
        ind2 = (i-1)%numbands
    else:
        ind2 = (i+1)%numbands
    
    uymax1 = uy[uymaxinds[0][ind2::numbands], uymaxinds[1][ind2::numbands]]
    xmax1 = x[uymaxinds[1][ind2::numbands]]
    
    uymax = (uymax0 + uymax1) / 2.0
    
    zonalamps[i, :] = uymin - uymax
    
    xmax2 = np.max(np.array([xmax0, xmax1]), axis=0)
    xmax3 = np.min(np.array([xmax0, xmax1]), axis=0)
    
    #print(xmax3[0], xmax2[0], xmin[0])
    
    if xmin[0] < xmax3[0] or xmin[0] > xmax2[0]:
        cutoff = (data['allxavgs'] <= xmax3[np.newaxis,:]) + (data['allxavgs'] >= xmax2[np.newaxis,:])
        zonallengths[i, :] = np.mod(xmax3 - xmax2, 2*np.pi)
    else:
        cutoff = (data['allxavgs'] >= xmax3[np.newaxis,:]) * (data['allxavgs'] <= xmax2[np.newaxis,:])
        zonallengths[i, :] = np.mod(xmax2 - xmax3, 2*np.pi)
        
    mixinglengths[i, :] = np.max(data['allxstds']*(data['allcorrdims']>1.5)*cutoff, axis=0)
    chaoticfraction[i, :] = np.sum((data['allcorrdims']>1.5)*cutoff, axis=0) / np.sum(cutoff, axis=0)
    
uybar = np.fft.irfft(np.fft.rfft(qbars['qbar'], axis=1)*kxinv, axis=1)


# %%
def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])
    
xsort = np.argsort(data['allxavgs'], axis=0)

numchaoticregions = np.zeros(uy.shape[0])
for i in range(uy.shape[0]):
    corrdims = data['allcorrdims'][xsort[:,i],i]
    z, p, ia = rle(corrdims > 1.5)
    
    numchaoticregions[i] = np.sum((z > 5) * ia)
    
# %% Contour data
    
#cdata = np.load('case1_snapcontours.npz')

# %%

plt.figure()
ax = plt.subplot(311)

#for i in range(numbands):
#    ax.plot(mixinglengths[i,:] / zonallengths[i,:], marker='.')
#    ax.plot(chaoticfraction[i,:], marker='.')
#ax.plot(mixinglengths[2,:] / zonallengths[2,:], marker='.', c='tab:orange')
#axt = ax.twinx()
#axt.plot(cdata['lenallcontours'][:,425])
#axt.plot(np.max(uy, axis=1) - np.min(uy, axis=1), c='tab:orange')
#axt.plot(zonalamps[1,:], c='tab:green')
#axt.plot(zonalamps[2,:], c='tab:red')
#axt.plot(np.max(ranresids, axis=0), c='tab:orange', marker='.')

ax.plot(np.average(data['allcorrdims']>1.5, axis=0))

ax2 = plt.subplot(312)

xsort = np.argsort(data['allxavgs'], axis=0)
xplot = np.take_along_axis(data['allxavgs'], xsort, axis=0)
corrplot = np.take_along_axis(data['allcorrdims'], xsort, axis=0)
trange = np.arange(0, 64.1, 0.25)
trange2 = np.arange(-0.125, 64.0 + 0.125 + 0.01, 0.25)
tplot = np.zeros(xplot.shape)
tplot[:,:] = trange[np.newaxis, :]

def recenter(z):
    z2 = np.zeros(z.shape[0]+1)
    z2[1:-1] = (z[1:] + z[:-1]) / 2.0
    z2[0] = 2*z[0] - z[1]
    z2[-1] = 2*z[-1] - z[-2]
    return z2

"""
tplot2 = recenter(tplot)
xplot2 = recenter(xplot.T).T

triangles = np.zeros((len(np.ravel(corrplot))*2, 3))
facecolors = np.zeros(len(np.ravel(corrplot))*2)

facecolors[::2] = np.ravel(corrplot)
facecolors[1::2] = np.ravel(corrplot)

for i in range(len(np.ravel(corrplot))):
    ind1 = i%corrplot.shape[1]
    ind0 = i//corrplot.shape[1]
    
    triangles[2*i] = [ind1+ind0*(corrplot.shape[1]+1), (ind1+1)+ind0*(corrplot.shape[1]+1), ind1+(ind0+1)*(corrplot.shape[1]+1)]
    triangles[2*i+1] = [(ind1+1)+ind0*(corrplot.shape[1]+1), (ind1+1)+(ind0+1)*(corrplot.shape[1]+1), ind1+(ind0+1)*(corrplot.shape[1]+1)]
"""

for i in range(len(trange)):
    xplot2 = recenter(xplot[:,i])
    tplot2 = np.array([trange2[i], trange2[i+1]])
    
    tplot3, xplot3 = np.meshgrid(tplot2, xplot2)
    
    ax2.pcolormesh(tplot3, xplot3, np.array([corrplot[:,i]]).T, vmin=1.0, vmax=2.0, shading='flat')

#ax2.pcolormesh(tplot, xplot, corrplot, vmin=1.0, vmax=2.0, shading='gouraud')
#ax2.scatter(np.ravel(tplot), np.ravel(data['allxavgs']), c=np.ravel(data['allcorrdims']), cmap='viridis', vmin=1.0, vmax=2.0, s=(72.0/100.0)**2)
#ax2.tripcolor(np.ravel(tplot2), np.ravel(xplot2), triangles, facecolors=facecolors)


ax3 = plt.subplot(313)
#ax3.pcolormesh(np.gradient(qbars['qbar'], axis=1).T / (2*np.pi/nx) < -8, cmap='PiYG')
#axt.plot(np.average(qbars['pvflux'][:,1600:1700], axis=1), c='tab:red', marker='.')
ax3.pcolormesh(cdata['lenallcontours'].T)

#plt.figure()
#plt.plot(mixinglengths[1,:], zonalamps[1,:], marker='.')

# %%

xsort = np.argsort(np.ravel(data['allxavgs']))
xorbit = np.average(np.reshape(np.ravel(data['allxavgs'])[xsort], data['allxavgs'].shape), axis=1)
chfraction = np.average(np.reshape(np.ravel(data['allcorrdims'])[xsort]>1.5, data['allxavgs'].shape), axis=1)

kuofraction = np.average((np.gradient(qbars['qbar'], axis=1) / (2*np.pi/nx))+8 < 0, axis=0)

#xsort = np.argsort(data['allxavgs'], axis=0)
#xorbit = np.average(np.take_along_axis(data['allxavgs'], xsort, axis=0), axis=1)
#chfraction = np.average(np.take_along_axis(data['allcorrdims'], xsort, axis=0)>1.5, axis=1)

plt.figure()
ax = plt.subplot(111)
ax.plot(x,np.gradient(np.average(qbars['qbar'], axis=0))/np.gradient(x)+8)
ax.axhline(0)

axt = ax.twinx()
axt.plot(xorbit, chfraction, c='tab:orange')
axt.plot(x, kuofraction, c='tab:green')