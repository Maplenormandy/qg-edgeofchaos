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

def poincarePlot(ax, pdata, colordata):
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
    axt = axm.twinx()
    
    axm.fill_between(xorbit, chfraction, color='tab:orange', fc=mpl.cm.tab20(0.175), lw=0)
    axm.plot(xorbit, chfraction, c='tab:orange')
    
    axm.set_ylim([0.0, 1.0])
    
    axt.axhline(8, c='k', ls='--')
    axt.plot(x, dqbar, c='tab:blue')
    
    axm.spines['left'].set_color('tab:blue')
    axt.spines['left'].set_color('tab:blue')
    
    axm.spines['right'].set_color('tab:orange')
    axt.spines['right'].set_color('tab:orange')
    
    axt.yaxis.label.set_color('tab:blue')
    axt.tick_params(axis='y', colors='tab:blue')
    
    axm.yaxis.label.set_color('tab:orange')
    axm.tick_params(axis='y', colors='tab:orange')
    
    if (case==1):
        axt.set_ylim([0.0, 40.0])
        axm.text(0.0, 1.02, r"$\bar{q}'(y)+\beta$", transform=axm.transAxes, ha='left', va='bottom', c='tab:blue')
        axm.text(1.0, 1.02, r'$f_{\mathrm{chaotic}}$', transform=axm.transAxes, ha='right', va='bottom', c='tab:orange')
    else:
        axt.set_ylim([0.0, 30.0])
    
    axm.yaxis.tick_right()
    axt.yaxis.tick_left()
    
    axm.set_xlim([-np.pi, np.pi])
    
    """
    ### Plotting for qtilde ###
    inputdata = np.load('../poincare_input/case{}_poincare_config_fd_smooth_uphavg.npz'.format(case))
    

    # Compute q for each eigenfunction    
    numeigs = inputdata['psiv'].shape[0]
    psiv = inputdata['psiv']
    kys = inputdata['kys']
    qv = np.zeros(inputdata['psiv'].shape)
    dx = 2*np.pi/nx
    cent_d2x = (np.diag(np.ones(nx-1), 1)+np.diag(np.ones(nx-1), -1) - 2*np.eye(nx) + np.diag(np.ones(1), -(nx-1))+np.diag(np.ones(1), (nx-1))) / dx**2

    for i in range(numeigs):
        ky = kys[i]
        lap = (cent_d2x - np.eye(nx)*(ky**2))
        qv[i,:] = lap @ psiv[i,:]
    
    # Compute qtilde
    camp = inputdata['amps'] * np.exp(1j*inputdata['phases'])
    qtildeplot = np.sqrt(np.sum(np.abs(camp[:, np.newaxis] * qv)**2, axis=0)/2.0)
    
    # Plot qtilde
    axq = axm.twinx()
    
    axq.plot(x, qtildeplot, ls='--', c='tab:green')
    axq.spines.right.set_position(("axes", 1.1))
    axq.yaxis.label.set_color('tab:green')
    axq.tick_params(axis='y', colors='tab:green')

    axq.spines['left'].set_color('tab:blue')
    axq.spines['right'].set_color('tab:green')
    """
    

# %% Plot figure

#fig = plt.figure(figsize=(17.8/2.54, 0.32*17.8/2.54), dpi=300)
#gs = fig.add_gridspec(1, 3, width_ratios=[2,2,2])


fig = plt.figure(figsize=(17.8/2.54*(2.0/3.0), 0.32*17.8/2.54), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[2,2])

#case = 1

for case in [1,2]:
    cdata = np.load('case{}_snapcontours.npz'.format(case))
    mdata = np.load('../poincare_analysis/case{}_mixing_lengths.npz'.format(case))
    qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))
    
    ### Load PV data ###
    #snapind = 51 if (case == 1) else 192
    snapind = 0 if (case == 1) else 192
    #snapind = 192
    snapfilenum = (snapind//16+2) if (case == 1) else (snapind//10+1)
    simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, snapfilenum), 'r')
    qindex = (snapind%16) if (case == 1) else (snapind%10)
    
    q = simdata['tasks/q'][qindex,:,:]
    
    lenq = circularInterpolant(cdata['levels'], np.average(cdata['lenmaxcontour'], axis=0), 2*8*np.pi, 500)
    
    
    ### Plot the poincare plots max amplitude ###
    pdata = np.load('../extra_poincare_sections/case{}_section_ind{:03d}_uphavg.npz'.format(case,snapind))
    axp1 = fig.add_subplot(gs[case-1])
    
    poincarePlot(axp1, pdata, mdata['allcorrdims'][:,snapind])
    
    
    axp1.set_xlabel('$x$')
    axp1.set_ylabel('$y$')
    #axp1.text(0.0, 1.05, '(a)', transform=axp1.transAxes, ha='left', va='bottom')
    axp1.set_title('Case {}'.format(case))








plt.tight_layout(h_pad=0.0, w_pad=1.5)
plt.tight_layout(h_pad=0.0, w_pad=1.5)
plt.margins(0, tight=True)

#plt.savefig('mixing_plot.pdf', dpi=300)
#plt.savefig('mixing_plot.png', dpi=300)