# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:33:46 2022

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal
from mpl_toolkits.axes_grid1 import make_axes_locatable

import h5py

import os
os.chdir('../')

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

# %% Setup helper info

def circularInterpolant(x, vec, ofs, pad):
    xp = np.zeros(len(x)+2*pad)
    xp[pad:-pad] = x
    xp[:pad] = x[-pad:] - ofs
    xp[-pad:] = x[:pad] + ofs
    
    
    vecp = np.zeros(len(x)+2*pad)
    vecp[pad:-pad] = vec
    vecp[:pad] = vec[-pad:]
    vecp[-pad:] = vec[:pad]

    return scipy.interpolate.interp1d(xp, vecp, kind='cubic')

nx = 2048
x = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)
kx = np.fft.rfftfreq(nx, 1.0/nx)
kxinv = np.zeros(kx.shape, dtype=complex)
kxinv[kx>0] = 1.0/(-1j*kx[kx>0])

def poincarePlot(ax, pdata, colordata, labelcax=True):
    z0 = pdata['y'][:,0]
    yclip = pdata['yclip']
    
    nparticles = len(z0)//2
    colors = np.zeros((nparticles, yclip.shape[1]))
    colors[:,:] = colordata[:, np.newaxis]
    ax.set_aspect('equal', adjustable='datalim')

    #ax.set_xticks([])
    sc = ax.scatter(yclip[nparticles:,:], yclip[:nparticles,:], s=(72.0/450.0)**2, marker='o', linewidths=0, c=colors[:,:], rasterized=True, cmap='viridis', vmin=1.0, vmax=2.0)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(sc, cax=cax)
    
    ax.set_xlim([-np.pi,np.pi])
    ax.set_ylim([-np.pi,np.pi])
    
    #ax.set_title('$d_{cor}$')
    
    if labelcax:
        cax.text(0.5, 1.05, '$d_C$', transform=cax.transAxes, ha='center', va='bottom')
    

def mixPlot(axm, case):
    mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))
    qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))
    
    # Compute things to plot
    qbar = np.average(qbars['qbar'], axis=0)
    dqbar = np.gradient(qbar) / np.gradient(x) + 8
    
    xsort = np.argsort(np.ravel(mdata['allxavgs']))
    xorbit = np.average(np.reshape(np.ravel(mdata['allxavgs'])[xsort], mdata['allxavgs'].shape), axis=1)
    chfraction = np.average(np.reshape(np.ravel(mdata['allcorrdims'])[xsort]>1.5, mdata['allxavgs'].shape), axis=1)
    
    kuofraction = np.average((np.gradient(qbars['qbar'], axis=1) / (2*np.pi/nx))+8 < 0, axis=0)
    
    # Plot of mixing
    axt = axm.twiny()
    
    axm.fill_betweenx(xorbit, chfraction, color='tab:orange', fc=mpl.cm.tab20(0.175), lw=0)
    axm.plot(chfraction, xorbit, c='tab:orange')
    
    axm.set_xlim([0.0, 1.0])
    
    axt.axvline(8, c='k', ls='--')
    axt.plot(dqbar, x, c='tab:blue')
    
    axm.spines['top'].set_color('tab:blue')
    axt.spines['top'].set_color('tab:blue')
    
    axm.spines['bottom'].set_color('tab:orange')
    axt.spines['bottom'].set_color('tab:orange')
    
    axt.xaxis.label.set_color('tab:blue')
    axt.tick_params(axis='x', colors='tab:blue')
    
    axm.xaxis.label.set_color('tab:orange')
    axm.tick_params(axis='x', colors='tab:orange')
    
    if (case==1):
        axt.set_xlim([0.0, 40.0])
        axt.set_xlabel(r"$\bar{q}'(y)+\beta$")
        #axm.text(0.0, 1.02, r"$\bar{q}'(y)+\beta$", transform=axm.transAxes, ha='left', va='bottom', c='tab:blue')
        #axm.text(1.0, 1.02, r'$f_{\mathrm{chaotic}}$', transform=axm.transAxes, ha='right', va='bottom', c='tab:orange')
    else:
        axt.set_xlim([0.0, 30.0])
        axm.set_xlabel(r'$f_{\mathrm{chaotic}}$')
    
    #axm.yaxis.tick_right()
    #axt.yaxis.tick_left()
    
    axm.set_ylim([-np.pi, np.pi])
    


# %% Plot figure

fig = plt.figure(figsize=(13.0/2.54, 13.0/2.54*0.73), dpi=100)
gs = fig.add_gridspec(2, 3, width_ratios=[2,2,1])


#fig = plt.figure(figsize=(17.8/2.54*(2.0/3.0), 0.32*17.8/2.54), dpi=300)
#gs = fig.add_gridspec(1, 2, width_ratios=[2,2])

for case in [1,2]:

    cdata = np.load('case{}_snapcontours.npz'.format(case))
    mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))
    qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))
    
    ### Load PV data ###
    #snapind = 51 if (case == 1) else 192
    snapind = 0 if (case == 1) else 192
    #snapind = 0
    snapfilenum = (snapind//16+2) if (case == 1) else (snapind//10+1)
    simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, snapfilenum), 'r')
    qindex = (snapind%16) if (case == 1) else (snapind%10)
    
    q = simdata['tasks/q'][qindex,:,:]
    
    lenq = circularInterpolant(cdata['levels'], np.average(cdata['lenmaxcontour'], axis=0), 2*8*np.pi, 500)
    
    
    ### Plot the poincare plots max amplitude ###
    pdata = np.load('../extra_poincare_sections/case{}_section_ind{:03d}_uphavg.npz'.format(case,snapind))
    axp1 = fig.add_subplot(gs[case-1, 0])
    
    if case == 1:
        poincarePlot(axp1, pdata, mdata['allcorrdims'][:,snapind])
    else:
        poincarePlot(axp1, pdata, mdata['allcorrdims'][:,snapind], labelcax=False)
    
    if case == 1:
        #axp1.set_title('Poincaré Section')
        axp1.text(0.0, 1.05, '(a)', transform=axp1.transAxes, ha='left', va='bottom')
    else:
        axp1.set_xlabel('$x$')
    axp1.set_ylabel('$y$')
    axp1.set_title('Case {}'.format(case))
    
    
    
    
    ### Plot the DNS snapshot ###
    axq = fig.add_subplot(gs[case-1, 1])
    im = axq.imshow(np.fliplr(lenq(q+8*x[:,np.newaxis])), origin='lower', cmap='viridis', extent=(-np.pi, np.pi, -np.pi, np.pi))
    divider = make_axes_locatable(axq)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(im, cax=cax)
    
    
    #axq.set_yticklabels([])
    
    if case == 1:
        #axq.set_title('DNS Snapshot')
        cax.text(0.5, 1.03, '$\ell_q$', transform=cax.transAxes, ha='center', va='bottom')
        axq.text(0.0, 1.05, '(b)', transform=axq.transAxes, ha='left', va='bottom')
    else:
        axq.set_xlabel('$x$')
    #axq.set_ylabel('$y$')
    
    
    ### Plot the boundaries of the chaotic region ###
    if case == 1:
        pcplot = [132, 186]
    else:
        pcplot = [61, 80, 143, 166] # snapind = 192
        #pcplot = [60, 81, 146, 166]
    #pcplot = [141, 172]
    yclip = pdata['yclip']
    
    for ind in pcplot:
        nparticles = yclip.shape[0]//2
        xind = np.argsort(yclip[nparticles+ind,:])
        xplot = (yclip[nparticles+ind,:])[xind]
        yplot = (yclip[ind,:])[xind]
        axp1.plot(xplot[::4], yplot[::4], c='tab:orange', ls='--', lw=0.9)
        axq.plot(xplot[::4], yplot[::4], c='tab:orange', ls='--', lw=0.9)
        pass
    
    
    
    
    ### Plot the PV gradient comparison ###
    
    axm = fig.add_subplot(gs[case-1,2])
    mixPlot(axm, case)
    
    if case == 1:
        axm.text(-0.15, 1.05, '(c)', transform=axm.transAxes, ha='left', va='bottom')
    
    
    #axm1.set_xticklabels([])
    #axm2.set_xlabel('$y$')
    
    #axm1.text(-np.pi+0.1, 1.0-0.05, 'Case 1', ha='left', va='top')
    #axm2.text(-np.pi+0.1, 1.0-0.05, 'Case 2', ha='left', va='top')
    
    #axm1.text(0.0, 1.2, '(c)', transform=axm1.transAxes, ha='left', va='bottom')



plt.tight_layout(pad=0, w_pad=2.0)
plt.tight_layout(pad=0, w_pad=2.0)

plt.savefig('plot_mixing.pdf', dpi=300)
#plt.savefig('mixing_plot.png', dpi=300)