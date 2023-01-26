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






cases = ['case1', 'case2']
suffixes = ['000', '001', '037']

poddatas = [[np.load('../../dns_input/{}/raw_podmode{}.npy'.format(cases[i], suffixes[j])) for i in range(len(cases))] for j in range(len(suffixes))]
# %%

fig = plt.figure(figsize=(13.0/2.54, 13.0*0.32/2.54), dpi=200)

gs_over = fig.add_gridspec(1, 5, width_ratios=[1,0.01,1, 0.01, 0.05])

ax_share = None

t = np.arange(64, step=0.25)

for i in range(len(cases)):
        
    gs = gs_over[2*i].subgridspec(2, 3, wspace=0.0, height_ratios=[0.75,0.25])
    timetraces = np.load('../../dns_input/{}/pod_timetraces.npz'.format(cases[i]))
    
    for j in range(len(suffixes)):
        ax0 = fig.add_subplot(gs[0, j])
        poddata = poddatas[j][i]
        
        im = ax0.imshow(np.fliplr(poddata), origin='lower', extent=(-np.pi,np.pi, -np.pi,np.pi))
        
        if ax_share == None:
            ax1 = fig.add_subplot(gs[1, j])
            ax_share = ax1
        else:
            ax1 = fig.add_subplot(gs[1, j], sharey=ax_share)
        
        ax1.axhline(ls=':', lw=0.4, c='gray')
        ax1.plot(t, timetraces['arr_0'][:,int(suffixes[j])], lw=0.4)
        ax1.set_ylim([-0.18,0.18])
        
        ax0.text(0.97, 0.97, 'Mode {}'.format(int(suffixes[j])), ha='right', va='top', transform=ax0.transAxes)
        
        ax0.set_xticks([-2,0,2])
        
        ax1.set_xticks([0, 20, 40, 60])
        if j == 0:
            ax0.xaxis.tick_top()
            
            if i == 0:
                ax1.text(0.03, 0.95, '$a(t)$', transform=ax1.transAxes, ha='left', va='top')
                
                #ax.xaxis.set_ticks_position('top')
                ax0.xaxis.set_label_position('top')
                ax0.set_xlabel('$x$')
                ax0.set_ylabel('$y$')
                ax1.set_xlabel('$t$')
                #ax0.text(0.03, 0.97, r'$\psi(x,y)$', transform=ax0.transAxes, ha='left', va='top')
                
                
            else:
                ax1.set_yticks([])
                ax0.set_xticklabels([])
                ax0.set_yticklabels([])
            #ax0.text(-0.03, 0.97, r'$y$', transform=ax0.transAxes, ha='right', va='top')
            #ax0.text(0.97, -0.03, r'$x$', transform=ax0.transAxes, ha='right', va='top')
        elif j == 1:
            
            ax0.set_title('Case {}'.format(i+1))
            
            ax1.set_yticks([])
            ax0.set_xticklabels([])
            ax0.set_yticklabels([])
            
        else:
            ax1.set_yticks([])
            ax0.set_xticklabels([])
            ax0.set_yticklabels([])

gs = gs_over[4].subgridspec(2, 1, wspace=0.0, height_ratios=[0.75,0.25])
ax = fig.add_subplot(gs[0])
cbar = fig.colorbar(im, cax=ax, orientation='vertical')
cbar.ax.text(0.5, 1.05, r'$\psi(x,y)$', ha='center', va='bottom', transform=cbar.ax.transAxes)
cbar.ax.get_yaxis().set_ticks([])

plt.tight_layout(pad=0.05)
plt.tight_layout(pad=0.05)

plt.savefig('plot_pod.pdf', dpi=300)
plt.savefig('plot_pod.png', dpi=600)

# %%

fig = plt.figure(figsize=(6.5/2.54, 6.5/2.54*0.45), dpi=300)

gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
#axspec = fig.add_subplot(gs[0])

cases = ['case1', 'case2']
labels = ['Case 1', 'Case 2']

for i in range(len(cases)):
    data = np.loadtxt('../../dns_input/{}/podsvals.txt'.format(cases[i]))
    cumenergy = np.cumsum(data**2)
    
    contrib = (cumenergy / cumenergy[-1])
    print(cumenergy[-1]/2048/2048/256)
    
    if i == 0:
        ax.semilogx(np.arange(len(contrib))+1, contrib, label=labels[i], marker='.')
        #axspec.loglog(np.arange(len(contrib))+1, data**2/cumenergy[-1], label=labels[i], marker='.')
    else:
        #axspec2 = axspec.twinx()
        ax.semilogx(np.arange(len(contrib))+1, contrib, label=labels[i], marker='.', ls=':')
        #axspec.loglog(np.arange(len(contrib))+1, data**2/cumenergy[-1], label=labels[i], marker='.', ls='--')

ax.set_xlabel('$n_{trunc}$')
ax.set_ylabel('$E_{n}/E_{tot}$')
#ax.set_title('(a)', loc='left')
ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=0.2))
ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=0.05))
ax.xaxis.grid(which='both', ls=':', lw=0.4)
ax.yaxis.grid(which='both', ls=':', lw=0.4)
[ax.spines[s].set_visible(False) for s in ax.spines]

plt.legend()

plt.tight_layout(pad=0)
plt.tight_layout(pad=0)

plt.savefig('plot_pod_spectrum.pdf', dpi=300)
plt.savefig('plot_pod_spectrum.png', dpi=600)