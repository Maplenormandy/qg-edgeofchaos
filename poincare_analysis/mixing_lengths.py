import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import scipy.interpolate
import scipy.signal

import scipy.spatial
import scipy.stats

import sys

# %% Load data


case = int(sys.argv[1])
print("Case " + str(case))
suffix = '_uphavg'
basedata = np.load('/home/nc1472/git/qg-edgeofchaos/poincare_input/case{}_poincare_config_fd_smooth_uphavg.npz'.format(case))

qbar = basedata['qbar']
uy = basedata['uy']

nx = 2048

x = np.linspace(-np.pi, np.pi, num=nx, endpoint=False)

# Set up interpolation functions
pad = 4
xp = np.zeros(nx+2*pad)
xp[pad:-pad] = x
xp[:pad] = x[-pad:] - 2*np.pi
xp[-pad:] = x[:pad] + 2*np.pi

def circularInterpolant(vec):
    vecp = np.zeros(nx+2*pad)
    vecp[pad:-pad] = vec
    vecp[:pad] = vec[-pad:]
    vecp[-pad:] = vec[:pad]

    return scipy.interpolate.interp1d(xp, vecp, kind='quadratic')

uyfft = np.fft.rfft(uy)
hilbuy = np.fft.irfft(1j*uyfft)
hilbuyf = circularInterpolant(hilbuy)
uyf = circularInterpolant(uy)

# Compute regions of zonal flow minima and maxima
uyminxs = x[scipy.signal.argrelextrema(uy, np.less)]
uymaxxs = x[scipy.signal.argrelextrema(uy, np.greater)]

# Set up function for computing correlation dimension
def fit_slope(lind, rind, psorted, bounds):
    lbound = bounds[lind]
    ubound = bounds[rind]

    sampinds = np.array(list(map(lambda x: int(np.round(x)), np.geomspace(lbound, ubound, num=256))), dtype=int)
    result = scipy.stats.linregress(np.log(psorted[sampinds-1]), np.log(ncorr[sampinds-1]))

    return result

# Set up result arrays
nparticles = 127
allstdresids = np.zeros((nparticles, 257))
allranresids = np.zeros((nparticles, 257))

allcorrdims = np.zeros((nparticles, 257))
allxavgs = np.zeros((nparticles, 257))
allxstds = np.zeros((nparticles, 257))
allrotnums = np.zeros((nparticles, 257))

for ind in range(257):
    print(ind)

    data = np.load('/data/nc1472/qg-edgeofchaos/extra_poincare_sections/case{}_section_ind{:03d}{}.npz'.format(case, ind, suffix), 'r')

    z0 = data['y'][:,0]
    yclip = data['yclip']
    yorig = data['y']

    nparticles = len(z0)//2
    colors = np.zeros((nparticles, yclip.shape[1]))

    rotation_number = (data['y'][nparticles:,-1] - data['y'][nparticles:,0]) / data['y'].shape[1] / 2 / np.pi
    xavg = np.average(data['y'][:nparticles,:], axis=1)
    xstd = np.std(data['y'][:nparticles,:], axis=1)

    # Compute "mixing lengths"
    stdresid = np.zeros(nparticles)
    rangeresid = np.zeros(nparticles)

    for i in range(nparticles):
        xall = data['y'][i,:] - xavg[i]

        nvar = 16

        ymat = np.zeros((nvar, len(xall)-nvar))
        xmat = np.zeros((nvar, len(xall)-nvar))

        for j in range(nvar):
            if j == 0:
                ymat[j,:] = xall[nvar-j:]
            else:
                ymat[j,:] = xall[nvar-j:-j]

            xmat[j,:] = xall[nvar-j-1:-(j+1)]

        amat = ymat @ np.linalg.pinv(xmat)
        residuals = ymat - (amat @ xmat)

        stdresid[i] = np.sqrt(np.average(np.abs(residuals[0,:])**2))
        rangeresid[i] = np.max(residuals[0,:]) - np.min(residuals[0,:])

    allstdresids[:,ind] = stdresid
    allranresids[:,ind] = rangeresid

    # Compute correlation dimensions
    corrdim = np.zeros(nparticles)

    for i in range(nparticles):
        sx = np.mod(yorig[i+nparticles,:], 2*np.pi)
        sy = yorig[i,:]

        sxd = np.mod(scipy.spatial.distance.pdist(np.array([sx]).T)+np.pi, 2*np.pi)-np.pi
        syd = scipy.spatial.distance.pdist(np.array([sy]).T)
        pdists = np.sqrt(sxd**2 + syd**2)
        psorted = np.sort(pdists)
        ncorr = np.arange(1, len(psorted)+1)

        bounds = list(map(lambda x: int(np.round(x)), np.geomspace(16, len(psorted+1), 32)))
        rsq = []
        slope = []

        lind = 0
        rind = len(bounds)-1

        result = fit_slope(lind, rind, psorted, bounds)

        rsq.append(result.rvalue**2)
        slope.append(result.slope)

        while rsq[-1] < 0.999 and (rind-lind)>16:
            resultl = fit_slope(lind+1, rind, psorted, bounds)
            resultr = fit_slope(lind, rind-1, psorted, bounds)

            if resultl.rvalue**2 > resultr.rvalue**2:
                lind = lind+1
                result = resultl
            else:
                rind = rind-1
                result = resultr

            rsq.append(result.rvalue**2)
            slope.append(result.slope)

        corrdim[i] = slope[-1]

    allcorrdims[:,ind] = corrdim
    allxavgs[:,ind] = xavg
    allxstds[:,ind] = xstd
    allrotnums[:,ind] = rotation_number

    np.savez('case{}_mixing_lengths.npz'.format(case), allcorrdims=allcorrdims, allxavgs=allxavgs, allxstds=allxstds, allrotnums=allrotnums, allstdresids=allstdresids, allranresids=allranresids)

