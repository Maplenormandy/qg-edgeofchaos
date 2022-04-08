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
mpl.rc('xtick', direction='in')
mpl.rc('ytick', direction='in')

mpl.rc('font', **font)

mpl.rc('mathtext', fontset='cm')

# %% POD Modes and time traces

fig = plt.figure(figsize=(8.7/2.54, 0.9*8.7/2.54), dpi=300)


gs_over = fig.add_gridspec(2, 1, height_ratios=[0.2, 0.5])

gs = gs_over[1].subgridspec(2, 3, hspace=0.0, wspace=0.0, height_ratios=[0.8,0.2])


ax = fig.add_subplot(gs_over[0])

cases = ['case1', 'case2']
labels = ['Case 1', 'Case 2']

for i in range(len(cases)):
    data = np.loadtxt('../dns_input/{}/podsvals.txt'.format(cases[i]))
    cumenergy = np.cumsum(data**2)
    
    contrib = (cumenergy / cumenergy[-1])
    
    ax.semilogx(np.arange(len(contrib))+1, contrib, label=labels[i], marker='.')

ax.set_xlabel('$n_{trunc}$')
ax.set_ylabel('$E_{n}/E_{tot}$')
ax.set_title('(a)', loc='left')
ax.xaxis.grid(which='major', ls=':', lw=0.4)
ax.yaxis.grid(which='both', ls=':', lw=0.4)




cases = ['case1']
suffixes = ['000', '001', '037']

ax_share = None

t = np.arange(64, step=0.25)

for i in range(len(cases)):
    timetraces = np.load('../dns_input/{}/pod_timetraces.npz'.format(cases[i]))
    
    for j in range(len(suffixes)):
        ax0 = fig.add_subplot(gs[0, j])
        poddata = np.load('../dns_input/{}/raw_podmode{}.npy'.format(cases[i], suffixes[j]))
        
        ax0.imshow(np.fliplr(poddata), origin='lower')
        ax0.set_xticks([])
        ax0.set_yticks([])
        
        if ax_share == None:
            ax1 = fig.add_subplot(gs[1, j])
            ax_share = ax1
        else:
            ax1 = fig.add_subplot(gs[1, j], sharey=ax_share)
        
        ax1.axhline(ls='--', lw=0.4, c='k')
        ax1.plot(t, timetraces['arr_0'][:,int(suffixes[j])], lw=0.4)
        ax1.set_ylim([-0.18,0.18])
        
        ax0.text(0.5, 1.0, 'Mode {}'.format(int(suffixes[j])), ha='center', va='bottom', transform=ax0.transAxes)
            
        if j == 0:
            ax1.text(0.03, 0.95, '$a(t)$', transform=ax1.transAxes, ha='left', va='top')
            ax0.set_title('(b)', loc='left')
            ax0.set_ylabel('$y$')
            ax0.text(0.03, 0.97, r'$\psi(x,y)$', transform=ax0.transAxes, ha='left', va='top')
        else:
            ax1.set_yticks([])
        
        if j == 1:
            ax1.set_xlabel('$t$')
            ax0.set_xlabel('$x$')


plt.tight_layout(h_pad=0.0, w_pad=0.0)
plt.tight_layout(h_pad=0.0, w_pad=0.0)

plt.savefig('pod_plots.pdf', dpi=1200)
plt.savefig('pod_plots.png', dpi=1200)