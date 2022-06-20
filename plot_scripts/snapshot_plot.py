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

fig = plt.figure(figsize=(17.8/2.54, 22.5/2.54), dpi=100)
gs = fig.add_gridspec(4, 2, height_ratios=[1.0,1.0,1.0,0.4])

ax = np.array([[fig.add_subplot(gs[i,j]) for j in range(2)] for i in range(4)])

plt.tight_layout()


for case in [1, 2]:
    snapind = 0
    snapfilenum = (snapind//16+2) if (case == 1) else (snapind//10+1)
    simdata = h5py.File('../dns_input/case{}/snapshots_s{}.h5'.format(case, snapfilenum), 'r')
    qindex = (snapind%16) if (case == 1) else (snapind%10)

    q = simdata['tasks/q'][qindex,:,:]
    
    # Compute various fields
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
    
    # Attempt to align the zonal color with the phase of q
    
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

    cfpsi = ax[0,case-1].imshow(np.fliplr(psitilde), origin='lower', extent=(-np.pi, np.pi, -np.pi, np.pi),
                             cmap='viridis', vmin=-psimax, vmax=psimax)
    caxpsi = fig.colorbar(cfpsi, ax=ax[0,case-1], orientation='vertical', label=r'$\tilde{\psi}$')
    
    cfq = ax[1,case-1].imshow(np.fliplr(qtilde), origin='lower', extent=(-np.pi, np.pi, -np.pi, np.pi),
                             cmap='PiYG', vmin=-qmax, vmax=qmax)
    fig.colorbar(cfq, ax=ax[1,case-1], orientation='vertical', label=r'$\tilde{q}$')
    
    caxq = ax[2,case-1].imshow(qplot_color, origin='lower', extent=(-np.pi, np.pi, -np.pi, np.pi))
    
    qnorm = mpl.colors.Normalize(vmin=(-8*np.pi-8*offset)/3.0, vmax=(8*np.pi-8*offset)/3.0)
    fig.colorbar(mpl.cm.ScalarMappable(norm=qnorm, cmap='twilight'),
                 ax=ax[2,case-1], orientation='vertical', extend='both', label=r'$q+\beta y$')
    #ax[2,case*2-1].axis('off')
    
    ax[2,case-1].set_xlabel('$x$')
    ax[2,case-1].set_ylabel('$y$')
    #caxpsi.ax.set_ylabel()
    
    ax[3,case-1].plot(x, vybar)
    ax[3,case-1].set_xlabel('$y$')
    ax[3,case-1].set_ylabel('$U(y)$')
    #ax[3,case*2-1].axis('off')
    
    ax[0,case-1].set_title('Case ' + str(case))
    
plt.tight_layout()
plt.tight_layout()

#plt.savefig('snapshot_plot.pdf', dpi=300)
#plt.savefig('snapshot_plot.png', dpi=300)