# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:38:19 2022

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt

# %%


suffix = '_uphavg'
case = 2

data = np.load('../poincare_input/case{}_poincare_config_fd_smooth{}.npz'.format(case, suffix))

qbar = np.copy(data['qbar'])

uy = np.copy(data['uy'])
psiv = np.copy(data['psiv'])
freq = np.copy(data['freq'])
freqs = np.copy(freq*data['freqmult'])
phases = np.copy(data['phases'])
kys = np.copy(data['kys'])
amps = np.copy(data['amps'])
numeigs = len(kys)

nx = psiv.shape[1]
kx = np.fft.rfftfreq(nx, 1.0/nx)
x = np.linspace(-np.pi,np.pi, num=nx, endpoint=False)

plt.figure()
espec = np.abs(np.fft.rfft(uy))**2
plt.loglog(kx, espec)
iref = 30
#plt.loglog(kx, (kx/kx[iref])**(-4) * espec[iref])
plt.loglog(kx, (kx/kx[iref])**(-5) * espec[iref])
#plt.loglog(kx, (kx/kx[iref])**(-6) * espec[iref])


# %% 

plt.figure()

