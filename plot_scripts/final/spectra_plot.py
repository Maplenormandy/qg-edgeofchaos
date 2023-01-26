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

data1 = np.load('../../dns_input/{}/qspec.npz'.format(cases[0]))
data2 = np.load('../../dns_input/{}/qspec.npz'.format(cases[1]))


# %% Compute 1d spectra


datas = [data1, data2]


fig = plt.figure(figsize=(13.0/2.54, 13.0*0.5/2.54), dpi=300)

for i in range(len(datas)):
    data = datas[i]
    
    

    qspec = data['qspec'] * (1+(ky>0)) /257/2048/2048/2048/2048 / 2
    espec = (qspec*(-invlap))
    
    print(np.sum(espec))
    
    pbqspec = np.real(data['pbqspec']) * (1+(ky>0)) /257/2048/2048/2048/2048
    pbespec = pbqspec*(-invlap)
    
    k2_1d = np.arange(1,1024)**2
    despec_1d = np.zeros(len(k2_1d))
    pbqspec_1d = np.zeros(len(k2_1d))
    pbespec_1d = np.zeros(len(k2_1d))
    
    espec_sorted = np.ravel(espec)[kinds]
    pbqspec_sorted = np.ravel(pbqspec)[kinds]
    pbespec_sorted = np.ravel(pbespec)[kinds]
    
    #intespec = np.cumsum(np.ravel(espec)[kinds])
    k2_1dinds = np.searchsorted(np.ravel(k2)[kinds], k2_1d, side='right')
    #espec_1d = intespec[k2_1dinds-1]
    
    despec_1d[0] = np.sum(espec_sorted[:k2_1dinds[0]])
    
    pbqspec_1d[0] = np.sum(pbqspec_sorted[:k2_1dinds[0]])
    pbespec_1d[0] = np.sum(pbespec_sorted[:k2_1dinds[0]])
    
    for j in range(1, len(k2_1d)):
        despec_1d[j] = np.sum(espec_sorted[k2_1dinds[j-1]:k2_1dinds[j]])
        pbqspec_1d[j] = np.sum(pbqspec_sorted[:k2_1dinds[j]])
        pbespec_1d[j] = np.sum(pbespec_sorted[:k2_1dinds[j]])
    #despec_1d = 
    k_1d = np.sqrt(k2_1d)-0.5
    
    #print(np.sum(espec))
    
    ax = plt.subplot(2, 2, i+1)
    plt.grid(ls=':')
    
    #ax.loglog(ky, 2*espec[:1025,0], label='zonal')
    #ax.loglog(ky, espec[0,:], label='meridional')
    
    iref = 14
        
    ax.axvspan(14, 15, fc='tab:red')
    ax.loglog(k_1d, despec_1d)
    #ax.loglog(k_1d, despec_1d, marker='.')
    #ax.loglog(k_1d, (k_1d/k_1d[iref])**(-2) * 2*espec[iref,0], label='-2', ls=':')
    #ax.loglog(k_1d, (k_1d/k_1d[iref])**(-3) * 2*espec[iref,0], label='-3', ls=':')
    ax.loglog(k_1d, (k_1d/k_1d[iref])**(-5.0/3.0) * despec_1d[iref], label='-5/3', ls=':')
    #ax.loglog(k_1d, (k_1d/k_1d[iref])**(-3) * despec_1d[iref], label='-3', ls=':')
    ax.loglog(k_1d, (k_1d/k_1d[iref])**(-4) * despec_1d[iref], label='-4', ls=':')
    #ax.loglog(k_1d, (k_1d/k_1d[iref])**(-5) * despec_1d[iref], label='-5')
    #ax.loglog(k_1d, (k_1d/k_1d[iref])**(-6.0) * despec_1d[iref], label='-6')
    
    #ax.loglog(k_1d, (k_1d/k_1d[iref])**(-3) * despec_1d[iref] * (np.log(k_1d/14) / np.log(k_1d[iref]/14))**(-1/3), label='-3', ls=':')

    prange2 = (k_1d/k_1d[iref])**(-5.0/3.0) * despec_1d[iref]
    plotrange = (k_1d/k_1d[iref])**(-4) * despec_1d[iref]
    
    ax.text(k_1d[500], prange2[500]*3, '$k^{-5/3}$')
    ax.text(k_1d[500], plotrange[500]*3, '$k^{-4}$')
    
    
    
    # Plot of flux
    ax2 = plt.subplot(2, 2, i+3)
    plt.grid(ls=':')
    
    ce = 'tab:blue'
    cq = 'tab:green'
    
    ax2.axvspan(14, 15, fc='tab:red')
    # energy flux
    mult = 1e3 if i==0 else 1e4
    ax2.semilogx(k_1d, pbespec_1d * mult, c=ce)
    
    ax3 = ax2.twinx()
    
    # enstrophy flux
    mult = 1e1 if i==0 else 1e2
    ax3.semilogx(k_1d, pbqspec_1d * mult, c=cq, ls='--')

    #ax.legend()
    ax.set_title('Case {}'.format(i+1))
    ax.set_ylabel('$E(k)$')
    ax.set_ylim((np.min(plotrange)/3, np.max(plotrange)*3))
    ax.set_xticklabels([])
    
    
    ax2.set_xlabel('$k$')
    
    
    ax2.yaxis.label.set_color(ce)
    ax2.tick_params(axis='y', colors=ce)
    
    ax3.yaxis.label.set_color(cq)
    ax3.tick_params(axis='y', colors=cq)
    
    ax2.spines['left'].set_color(ce)
    ax3.spines['left'].set_color(ce)
    
    ax2.spines['right'].set_color(cq)
    ax3.spines['right'].set_color(cq)
    
    
    if i == 0:
        ax2.set_ylim(np.array([-1*1.3, 1*1.3])*2)
        ax3.set_ylim(np.array([-2*1.3, 2*1.3])*2)
        ax2.set_yticks(np.array([-1, -0.5, 0.0, 0.5, 1])*2)
        ax3.set_yticks(np.array([-2, -1, 0.0, 1, 2])*2)
        
        ax2.set_ylabel(r'$\Pi_{E}(k) \; \left[\times 10^{-3}\right]$')
        ax3.set_ylabel(r'$\Pi_{Q}(k) \; \left[\times 10^{-1}\right]$')
    else:
        ax2.set_ylim(np.array([-0.5*1.3, 0.5*1.3])*2)
        ax3.set_ylim(np.array([-1*1.3, 1*1.3])*2)
        ax2.set_yticks(np.array([-0.5, -0.25, 0.0, 0.25, 0.5])*2)
        ax3.set_yticks(np.array([-1, -0.5, 0.0, 0.5, 1])*2)
        
        ax2.set_ylabel(r'$\Pi_{E}(k) \; \left[\times 10^{-4}\right]$')
        ax3.set_ylabel(r'$\Pi_{Q}(k) \; \left[\times 10^{-2}\right]$')
        
    #ax2.ticklabel_format(style='scientific',scilimits=(-1,1),axis='y')
    #ax3.ticklabel_format(style='scientific',scilimits=(-1,1),axis='y')

#plt.suptitle('Fourier Spectra')
plt.tight_layout(h_pad=0.0)
plt.tight_layout(h_pad=0.0)


plt.savefig('plot_kspectra.pdf', dpi=300)
plt.savefig('plot_kspectra.png', dpi=600)
