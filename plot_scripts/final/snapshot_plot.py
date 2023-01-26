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

# %%


fig = plt.figure(figsize=(13.0/2.54, 13.0/2.54*0.34), dpi=300)
gs = fig.add_gridspec(1, 4, width_ratios=[4,1,4,1])

ax = np.array([fig.add_subplot(gs[i]) for i in range(4)])

plt.tight_layout()


for case in [1, 2]:
    snapind = 0 if (case == 1) else 192
    snapfilenum = (snapind//16+2) if (case == 1) else (snapind//10+1)
    simdata = h5py.File('../../dns_input/case{}/snapshots_s{}.h5'.format(case, snapfilenum), 'r')
    qindex = (snapind%16) if (case == 1) else (snapind%10)

    q = simdata['tasks/q'][qindex,:,:]
    
    ### Compute various fields ###
    qfft = np.fft.rfft2(q)
    psifft = invlap*qfft
    vxfft = 1j*kyg*psifft
    vyfft = -1j*kxg*psifft
    psi = np.fft.irfft2(psifft)
    vx = np.fft.irfft2(vxfft)
    vy = np.fft.irfft2(vyfft)
    
    print(np.average(vx**2+vy**2))

    # Compute zonally-averaged quantities
    vybar = np.average(vy, axis=1)

    psibar = np.average(psi, axis=1)
    psitilde = psi-psibar[:,np.newaxis]
    psimax = np.max(np.abs(psitilde))

    qbar = np.average(q, axis=1)
    qtilde = q-qbar[:,np.newaxis]
    qmax = np.max(np.abs(qtilde))
    
    ### Attempt to align the zonal color with the phase of q ###
    
    if case == 1:
        numzones = 3
        offset = np.angle(qfft[numzones,0]) + 3.0*np.pi/4.0
        #offset = np.angle(qfft[numzones,0])
    else:
        numzones = 3
        offset = np.angle(qfft[numzones,0]) - np.pi/2.0
    
    if offset>np.pi:
        offset = offset-np.pi
        
    qplot = np.fliplr(q+8.0*(x[:,np.newaxis]+offset))
    qplot_clip = np.mod(numzones*qplot/(2*8*np.pi), 1.0)
    qplot_color = mpl.cm.twilight(qplot_clip)

    ### Plot stuff ###
    """
    cfpsi = ax[0,case-1].imshow(np.fliplr(psitilde), origin='lower', extent=(-np.pi, np.pi, -np.pi, np.pi),
                             cmap='viridis', vmin=-psimax, vmax=psimax)
    caxpsi = fig.colorbar(cfpsi, ax=ax[0,case-1], orientation='vertical', label=r'$\tilde{\psi}$')
    
    cfq = ax[1,case-1].imshow(np.fliplr(qtilde), origin='lower', extent=(-np.pi, np.pi, -np.pi, np.pi),
                             cmap='PiYG', vmin=-qmax, vmax=qmax)
    fig.colorbar(cfq, ax=ax[1,case-1], orientation='vertical', label=r'$\tilde{q}$')
    """
    
    # Plot q snapshot
    axq = ax[2*case-2]
    caxq = axq.imshow(qplot_color, origin='lower', extent=(-np.pi, np.pi, -np.pi, np.pi))
    
    qrange_color = mpl.cm.twilight(np.mod(numzones*np.linspace(np.min(qplot), np.max(qplot), num=256)/(2*8*np.pi), 1.0))
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('twilight2', qrange_color)
    qnorm = mpl.colors.Normalize(vmin=np.min(qplot)-8.0*offset, vmax=np.max(qplot)-8.0*offset)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=qnorm, cmap=cmap2),
                 ax=axq, orientation='vertical')
    cbar.ax.text(0.5, -0.1, r'$q+\beta y$', ha='center', va='bottom', transform=cbar.ax.transAxes)
    
    
    #ax[2,case*2-1].axis('off')
    
    axq.set_xlabel('$x$')
    if case == 1:
        axq.set_ylabel('$y$')
    #caxpsi.ax.set_ylabel()
    
    axu = ax[2*case-1]
    axu.axvline(ls=':', lw=0.4, c='gray')
    axu.plot(vybar, x)
    #axu.set_ylabel('$y$')
    axu.set_xlabel('$U(y)$')
    axu.set_ylim([-np.pi, np.pi])
    #ax[3,case*2-1].axis('off')
    
    plt.figtext(0.5*case - 0.25, 1.0, 'Case {}'.format(case), ha='center', va='top')
    
plt.tight_layout(pad=0, rect=(0.0,0.0, 1.0,0.92))
plt.tight_layout(pad=0, rect=(0.0,0.0, 1.0,0.92))


plt.savefig('plot_snapshot.pdf', dpi=300)
plt.savefig('plot_snapshot.png', dpi=600)