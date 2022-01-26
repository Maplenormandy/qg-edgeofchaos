import numpy as np
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from chm_utils import EigenvalueSolverFD

# %% Load basic state

print("Starting Script")
qbar = np.loadtxt('../qbar_equiv.txt')

x = np.linspace(-np.pi, np.pi, num=len(qbar), endpoint=False)

smoother = np.exp(-96*(x+np.pi)) + np.exp(-96*(np.pi-x))
smoother = smoother / np.sum(smoother)

qbar_smooth = np.fft.irfft(np.fft.rfft(qbar)*np.fft.rfft(smoother))

print(len(qbar))
print("Constructing")

eigsolver = EigenvalueSolverFD(qbar_smooth)
nky = 8
rangeky = range(1,1+nky)
eigs = [None]*nky

print("Solving for eigenfunctions")
for ky in rangeky:
    print(ky)
    eigs[ky-1] = eigsolver.solveEigenfunctions(ky=ky, norm='action')

numsnaps = 257
snapindex = 0

amps = np.zeros((nky, numsnaps, 2048), dtype=np.complex)

numfiles = numsnaps//10

print("Processing Data")
for s in range(1,numfiles+2):
    with h5py.File('../../snapshots/snapshots_s{}.h5'.format(s), mode='r') as simdata:
        qall = simdata['tasks/q']
        print('../../snapshots/snapshots_s{}.h5'.format(s))

        for index in range(qall.shape[0]):
            if (snapindex%4) == 0:
                print(snapindex)

            qffty = np.fft.rfft(qall[index,:,:], axis=1)
            for ky in rangeky:
                amps[ky-1,snapindex,:] = np.conj(eigs[ky-1]['vl']).T @ qffty[:,ky]

            snapindex = snapindex+1

np.savez('amps_fd_smooth.npz', amps=amps, qbar=qbar_smooth)
