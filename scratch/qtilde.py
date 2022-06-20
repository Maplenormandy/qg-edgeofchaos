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

x = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)


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



# %%
            

plt.figure()

axm = plt.subplot(111)

# Compute things to plot
qbar = np.average(qbars['qbar'], axis=0)
dqbar = np.gradient(qbar) / np.gradient(x) + 8

xsort = np.argsort(np.ravel(mdata['allxavgs']))
xorbit = np.average(np.reshape(np.ravel(mdata['allxavgs'])[xsort], mdata['allxavgs'].shape), axis=1)
chfraction = np.average(np.reshape(np.ravel(mdata['allcorrdims'])[xsort]>1.5, mdata['allxavgs'].shape), axis=1)

kuofraction = np.average((np.gradient(qbars['qbar'], axis=1) / (2*np.pi/nx))+8 < 0, axis=0)

# Plot of mixing
axt = axm.twinx()
axq = axm.twinx()

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

camp = inputdata['amps'] * np.exp(1j*inputdata['phases'])
qtildeplot = np.sqrt(np.sum(np.abs(camp[:, np.newaxis] * qv)**2, axis=0)/2.0)

axq.plot(x, qtildeplot, ls='--', c='tab:green')

axm.yaxis.tick_right()
axt.yaxis.tick_left()

if (case==1):
    axt.set_ylim([0.0, 40.0])
    axq.set_ylim([0.0, 4.0])
    axm.text(0.0, 1.02, r"$\bar{q}'(y)+\beta$", transform=axm.transAxes, ha='left', va='bottom', c='tab:blue')
    axm.text(1.0, 1.02, r'$f_{\mathrm{chaotic}}$', transform=axm.transAxes, ha='right', va='bottom', c='tab:orange')
else:
    axt.set_ylim([0.0, 30.0])

axm.set_xlim([-np.pi, np.pi])

#axq.yaxis.tick_left()
axq.spines.right.set_position(("axes", 1.1))
axq.yaxis.label.set_color('tab:green')
axq.tick_params(axis='y', colors='tab:green')

axq.spines['left'].set_color('tab:blue')
axq.spines['right'].set_color('tab:green')

plt.tight_layout()

# %%

qtilde = np.abs(camp[:, np.newaxis]) * qv
ampsort = np.argsort(-np.abs(camp))

plt.figure()
plt.axhline(ls='--', c='k')
for i in range(5):
    plt.plot(x, qtilde[ampsort[i],:])