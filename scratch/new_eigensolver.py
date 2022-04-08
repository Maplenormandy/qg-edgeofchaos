# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:18:05 2022

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import scipy.linalg

# %%

case = 2

nx = 2048
x = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)
xf = np.linspace(-np.pi, np.pi, num=nx*2, endpoint=False)

qbars = np.load('../dns_input/case{}/qbars.npz'.format(case))

qbar = qbars['qequiv'][0,:] - 8*x

# %%

smoother = np.exp(-(x+np.pi)**2/2*64**2) + np.exp(-(np.pi-x)**2/2*64**2)
smoother = smoother / np.sum(smoother)

qbar_smooth = np.fft.irfft(np.fft.rfft(qbar)*np.fft.rfft(smoother))

fig = plt.figure()

ax = plt.subplot(311)
ax.plot(qbar)
ax.plot(qbar_smooth)

ax = plt.subplot(312)
ax.plot(np.gradient(qbar)/(2*np.pi/nx))
ax.plot(np.gradient(qbar_smooth)/(2*np.pi/nx))

ax2 = plt.subplot(313)

ax2.semilogy(np.abs(np.fft.rfft(qbar)))
ax2.semilogy(np.abs(np.fft.rfft(qbar_smooth)))

# %% Interpolator

# Coarse realspace identity operator I
ec = np.eye(nx)
# FFT * I
fc = np.fft.rfft(ec, axis=0, norm='forward')
# Refinement * FFT * I
rfc = np.zeros((nx+1, nx), dtype=complex)
rfc[:nx//2+1,:] = fc
# IFFT * Refinement * FFT * I
refine = np.fft.irfft(rfc, axis=0, norm='forward')

# Fine realspace identity operator I
er = np.eye(nx*2)
# FFT * I
fr = np.fft.rfft(er, axis=0, norm='backward')
# Coarsening * FFT * I
cfr = np.zeros((nx//2+1, nx*2), dtype=complex)
cfr[:,:] = fr[:nx//2+1,:]
# IFFT * Coarsening * FFT * I
coarsen = np.fft.irfft(cfr, axis=0, norm='backward')

# %%

def fourierRefine(f):
    fc = np.fft.rfft(f, norm='forward')
    ff = np.zeros((nx+1), dtype=complex)
    ff[:nx//2+1] = fc
    return np.fft.irfft(ff, norm='forward')

def fourierCoarsen(f):
    ff = np.fft.rfft(f, norm='backward')
    fc = np.zeros((nx//2+1), dtype=complex)
    fc = ff[:nx//2+1]
    return np.fft.irfft(fc, norm='backward')

beta = 8.0
kx = np.fft.rfftfreq(nx, 1.0/nx)
kxinv = np.zeros(kx.shape, dtype=complex)
kxinv[kx>0] = 1.0/(-1j*kx[kx>0])
uy = np.fft.irfft(np.fft.rfft(qbar)*kxinv)
uypp = np.fft.irfft(-1j*kx*np.fft.rfft(qbar_smooth))
b = -beta + uypp

kxf = np.fft.rfftfreq(nx*2, 0.5/nx)

# %%

kd = 0
ky = 1
greenf = -np.fft.rfft(np.eye(nx*2), axis=0, norm='ortho') / (kxf**2 + kd**2 + ky**2)[:, np.newaxis]
green = np.fft.irfft(greenf, axis=0, norm='ortho')


msqrtb = np.diag(np.sqrt(fourierRefine(-b)))

pbq = (coarsen @ msqrtb @ green @ msqrtb @ refine)*(1/2)
mu = -(coarsen @ np.diag(fourierRefine(uy)) @ refine)*(1/2)

lu = -mu + pbq
luh = (lu + lu.T)/2.0

w, vh = scipy.linalg.eigh(luh)

# %%
plt.scatter(w, np.zeros(len(w)))

# %%

eig = 411
plt.figure()

ax = plt.subplot(111)
ax.plot(x, vh[:,eig])

axt = ax.twinx()
axt.plot(x, uy, c='tab:orange')


# %%
plt.figure()
plt.imshow(np.clip(vh, -np.sqrt(1/1024), np.sqrt(1/1024)), cmap='PiYG')

# %%



#plt.figure()
#plt.plot(x, qbar_smooth)
#plt.plot(xf, fourierRefine(qbar_smooth))
#plt.plot(x, fourierCoarsen(fourierRefine(qbar_smooth))*(1/2))


#plt.plot(x, uy*uy)
#plt.plot(x, mu @ uy)


#plt.imshow(refine.T-coarsen)
#plt.colorbar()