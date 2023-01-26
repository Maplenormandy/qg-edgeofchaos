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

data1 = np.load('../dns_input/{}/qspec.npz'.format(cases[0]))
data2 = np.load('../dns_input/{}/qspec.npz'.format(cases[1]))


# %% Compute 1d spectra


data = data2

qspec = data['qspec'] * (1+(ky>0)) /257
#espec = (qspec*(-invlap))

qfft = (np.random.randn(*qspec.shape) + 1j*np.random.randn(*qspec.shape)) * np.sqrt(qspec)
qfft[1:1024,0] = np.conj(qfft[2048:1024:-1,0])
qtest = np.fft.irfft2(qfft)

# %%

plt.imshow(qtest+8*x[:,np.newaxis], origin='lower')