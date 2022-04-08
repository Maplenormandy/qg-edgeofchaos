# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 17:17:45 2022

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

# %%
utildey = np.fft.irfft(-1j*kx[np.newaxis,:]*np.fft.rfft(psiv, axis=1), axis=1)
qtilde = np.fft.irfft(-(kx[np.newaxis,:]**2 + kys[:,np.newaxis]**2)*np.fft.rfft(psiv, axis=1), axis=1)


psiv1 = np.roll(psiv, 1, axis=1)
psiv2 = np.roll(psiv, -1, axis=1)

utildey = (psiv1 - psiv2) / (2 * np.pi / nx * 2)
qtilde = (psiv1 + psiv2 - 2*psiv) / (2 * np.pi / nx)**2 - kys[:,np.newaxis]**2 * psiv

# %%
plt.plot(qtilde2[0,:]-qtilde[0,:])
#plt.plot()
