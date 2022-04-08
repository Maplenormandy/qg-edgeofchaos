# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:51:47 2022

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.interpolate
import scipy.signal

import sys, os
sys.path.append(os.path.abspath('../'))

from poincare_map import PoincareMapper

# %%

suffix = '_uphavg'
#suffix = ''
case = 1
pm = PoincareMapper('../poincare_input/case{}_poincare_config_fd_smooth{}.npz'.format(case, suffix))
numeigs = len(pm.data['kys'])

timedata = np.load('../poincare_input/case{}_eigencomponent_timedata{}.npz'.format(case, suffix), 'r')


# %%

ind = 51
nparticles = 197
zonalmult = 1.0

ampmult = timedata['ampdevs'][:,ind]
phaseoffs = timedata['phasedevs'][:,ind]

data = pm.data

qbar = data['qbar']

uy = data['uy']
psiv = data['psiv']
freq = data['freq']
freqs = freq*data['freqmult']
phases = data['phases']
kys = data['kys']
amps = data['amps']
numeigs = len(kys)

psif, utyf, qtf, uyf = pm.funcs
x = pm.x
nx=len(x)

amps_mod = amps * ampmult
phases_mod = phases + phaseoffs

# Compute desried contour
qts = np.array(list((qtf[i](x)[:,np.newaxis])*(np.cos(kys[i]*x - phases_mod[i])[np.newaxis,:])*amps_mod[i] for i in range(numeigs)))

qtilde = np.sum(qts, axis=0)
qtilderadial = np.sum(qtilde**2, axis=0)

qbarf = pm.qbarf
xsamples = np.linspace(-np.pi, np.pi, num=nparticles, endpoint=False)
qsamples = qbarf(xsamples) + 8*xsamples

# The x direction of the minimum disruption to the q contours
qminampind = np.argmin(qtilderadial)

# The q to use to place the points
qtest = np.sort(qbar*zonalmult + 8*x + qtilde[:,qminampind])

qtest2 = np.sort(np.concatenate((qtest-8*2*np.pi, qtest, qtest+8*2*np.pi)))
x2 = np.concatenate((x-2*np.pi, x, x+2*np.pi))

xfunc = scipy.interpolate.interp1d(qtest2, x2)

x0 = xfunc(qsamples)
