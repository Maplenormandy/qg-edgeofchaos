import numpy as np
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

nx = 2048
numsnaps = 129
numfiles = numsnaps//16

qbar = np.zeros((numsnaps, nx))
qbar_equiv = np.zeros((numsnaps, nx))

x = np.linspace(-np.pi,np.pi, num=nx, endpoint=False)

timeindex = 0
snapindex = 0


print("Loading Data")

for s in range(1,numfiles+2):
    with h5py.File('../snapshots/snapshots_s{}.h5'.format(s), mode='r') as simdata:
        qall = simdata['tasks/q']
        print('../snapshots/snapshots_s{}.h5'.format(s))

        for index in range(qall.shape[0]):
            if (snapindex%4) == 0:
                print(snapindex)

            q = qall[index,:,:]

            qbar[snapindex,:] = np.average(q, axis=1)

            qtotal = q + 8*x[:,np.newaxis]
            qsorted = np.sort(np.ravel(qtotal))
            excessarea = qsorted - np.flip(qsorted) + 8*2*np.pi
            ind0 = np.searchsorted(excessarea, 0)
            qsorted2 = np.sort(np.mod(qsorted-qsorted[ind0], 8*2*np.pi))+qsorted[ind0]
            qbar_equiv[snapindex,:] = qsorted2[::2048]
            snapindex = snapindex+1


qallsorted = np.sort(np.ravel(qbar_equiv))
excessareaall = qallsorted - np.flip(qallsorted) + 2*8*np.pi
ind1 = np.searchsorted(excessareaall, 0)
qallsorted2 = np.sort(np.mod(qallsorted-qallsorted[ind1], 8*2*np.pi))+qallsorted[ind1]

np.savetxt('qbar.txt', np.average(qbar, axis=0))
np.savetxt('qbar_equiv.txt', qallsorted2[::numsnaps]-8*x)
