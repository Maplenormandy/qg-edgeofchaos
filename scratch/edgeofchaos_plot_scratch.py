# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 10:29:44 2022

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# %%

case = 2

amps = np.arange(0.1, 1.61, 0.1)

adata = np.load('../poincare_analysis/case{}_mixing_lengths_amps.npz'.format(case))
mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))

timedata = np.load('../poincare_input/case{}_eigencomponent_timedata_uphavg.npz'.format(case))
inputdata = np.load('../poincare_input/case{}_poincare_config_fd_smooth_uphavg.npz'.format(case))

qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))

# Compute the energy tensor

nx = 2048
numeigs = inputdata['psiv'].shape[0]

etensor = np.zeros((numeigs, numeigs))

psiv = inputdata['psiv']
kys = inputdata['kys']
qv = np.zeros(inputdata['psiv'].shape)
dx = 2*np.pi/nx
cent_d2x = (np.diag(np.ones(nx-1), 1)+np.diag(np.ones(nx-1), -1) - 2*np.eye(nx) + np.diag(np.ones(1), -(nx-1))+np.diag(np.ones(1), (nx-1))) / dx**2

for i in range(numeigs):
    ky = kys[i]
    lap = (cent_d2x - np.eye(nx)*(ky**2))
    qv[i,:] = lap @ psiv[i,:]
    
for i in range(numeigs):
    for j in range(i,numeigs):
        if kys[i] != kys[j]:
            pass
        else:
            etensor[i,j] = -psiv[i,:] @ qv[j,:]
            etensor[j,i] = -psiv[j,:] @ qv[i,:]


# %%
            

plt.figure()

chaoticfractions = np.average(mdata['allcorrdims']>1.5, axis=0)

quartiles = np.quantile(chaoticfractions, [0.0, 0.25, 0.75, 1.0])

colors = np.linspace(0.0, 1.0, num=20) + (1.0/40.0)

zamps = inputdata['amps']*np.exp(1j*inputdata['phases'])


avgenergy = np.real(np.conj(zamps) @ (etensor @ zamps))
snapenergies = np.zeros(timedata['ampdevs'].shape[1])

for i in range(timedata['ampdevs'].shape[1]):
    zamps = inputdata['amps']*timedata['ampdevs'][:,i]*np.exp(1j*(inputdata['phases']+timedata['phasedevs'][:,i]))
    snapenergies[i] = np.real(np.conj(zamps) @ (etensor @ zamps))


#avgamp = np.sum(inputdata['amps']**2)
#ampdevs = np.sqrt(np.sum((inputdata['amps'][:,np.newaxis] * timedata['ampdevs'])**2, axis=0) / avgamp)

ax = plt.subplot(111)

#plt.axhspan(quartiles[0], quartiles[3], fc=mpl.cm.tab20c(colors[7]), lw=0)
#plt.axhspan(quartiles[1], quartiles[2], fc=mpl.cm.tab20c(colors[6]), lw=0)
#ax.plot(amps, np.max((adata['allcorrdims']>1.5)*(adata['allxstds']), axis=0))


#rotminind = np.argmin(mdata['allrotnums'], axis=0)
#ax.scatter(mdata['allrotnums'][rotminind, np.arange(257)], mdata['allcorrdims'][rotminind, np.arange(257)])

#ax.axvline(1.0, c='k', ls='--')

#axt = ax.twinx()
#axt.hist(ampdevs, bins=16)

#ax.hist(chaoticfractions, bins=16)
#
#axt = ax.twinx()
#
#axt.plot(np.average(adata['allcorrdims']>1.5, axis=0), amps)

#plt.scatter(np.sqrt(snapenergies/avgenergy), chaoticfractions)
#plt.plot(amps, np.average((adata['allcorrdims']>1.5), axis=0))

#plt.plot(np.sort(chaoticfractions))
#plt.hist(chaoticfractions, bins=16)

plt.hist(chaoticfractions)
