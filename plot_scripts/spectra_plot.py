# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:11:30 2022

@author: maple
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl

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



nx = 2048

x = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)
y = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)

kx = np.fft.fftfreq(nx, 1.0/nx)
ky = np.fft.rfftfreq(nx, 1.0/nx)


kxg, kyg = np.meshgrid(kx, ky, indexing='ij')
xg, yg = np.meshgrid(x, y, indexing='ij')

k2 = kxg**2 + kyg**2
invlap = np.zeros(k2.shape)
invlap[k2>0] = -1.0 / k2[k2>0]

kinds = np.argsort(np.ravel(k2))



cases = ['case1', 'case2']

data1 = np.load('../dns_input/{}/qspec.npz'.format(cases[0]))
data2 = np.load('../dns_input/{}/qspec.npz'.format(cases[1]))


# %% Compute 1d spectra


datas = [data1, data2]

plt.figure()


for i in range(len(datas)):
    data = datas[i]
    
    

    qspec = data['qspec'] * (1+(ky>0)) /257/2048/2048/2048/2048
    espec = (qspec*(-invlap))
    
    k2_1d = np.arange(1,1024)**2
    despec_1d = np.zeros(len(k2_1d))
    
    espec_sorted = np.ravel(espec)[kinds]
    
    #intespec = np.cumsum(np.ravel(espec)[kinds])
    k2_1dinds = np.searchsorted(np.ravel(k2)[kinds], k2_1d, side='right')
    #espec_1d = intespec[k2_1dinds-1]
    
    despec_1d[0] = np.sum(espec_sorted[:k2_1dinds[0]])
    for j in range(1, len(k2_1d)):
        despec_1d[j] = np.sum(espec_sorted[k2_1dinds[j-1]:k2_1dinds[j]])
    #despec_1d = 
    k_1d = np.sqrt(k2_1d)-0.5
    
    #print(np.sum(espec))
    
    ax = plt.subplot(1, 2, i+1)
    plt.grid(ls=':')
    
    #ax.loglog(ky, 2*espec[:1025,0], label='zonal')
    #ax.loglog(ky, espec[0,:], label='meridional')
    
    iref = 14
        
    ax.loglog(k_1d, despec_1d, marker='.')
    #ax.loglog(k_1d, despec_1d, marker='.')
    #ax.loglog(k_1d, (k_1d/k_1d[iref])**(-2) * 2*espec[iref,0], label='-2', ls=':')
    #ax.loglog(k_1d, (k_1d/k_1d[iref])**(-3) * 2*espec[iref,0], label='-3', ls=':')
    ax.loglog(k_1d, (k_1d/k_1d[iref])**(-5.0/3.0) * despec_1d[iref], label='-5/3', ls=':')
    #ax.loglog(k_1d, (k_1d/k_1d[iref])**(-3) * despec_1d[iref], label='-3', ls=':')
    ax.axvspan(14, 15)
    ax.loglog(k_1d, (k_1d/k_1d[iref])**(-4) * despec_1d[iref], label='-4', ls=':')
    #ax.loglog(k_1d, (k_1d/k_1d[iref])**(-5) * despec_1d[iref], label='-5')
    #ax.loglog(k_1d, (k_1d/k_1d[iref])**(-6.0) * despec_1d[iref], label='-6')

    ax.legend()

plt.suptitle('Total')
