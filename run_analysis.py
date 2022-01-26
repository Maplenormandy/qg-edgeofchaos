# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:40:57 2021

@author: maple
"""

import numpy as np
from poincare_map import PoincareMapper

# %% Set up Baseline Poincare Mapper

#pm = PoincareMapper('poincare_input/poincare_config_validation.npz')
#numeigs = len(pm.data['kys'])

#pm.generateFullSection(np.ones(numeigs), np.zeros(numeigs), 'sections/section_validation.npz')



# %% Set up Poincare Mapper

pm = PoincareMapper('poincare_input/poincare_config_fd_smooth.npz')
numeigs = len(pm.data['kys'])


# %% Generate some sections of breaking waves

#pm.generateBreakingSection(np.ones(numeigs)*0.4, np.zeros(numeigs), pm.qmin, 16, 'sections/breaking_amp040.npz')
#pm.generateBreakingSection(np.ones(numeigs)*1.0, np.zeros(numeigs), pm.qmin, 16, 'sections/breaking_amp100.npz')
#pm.generateBreakingSection(np.ones(numeigs)*1.2, np.zeros(numeigs), pm.qmin, 16, 'sections/breaking_amp120.npz')

# %% Generate amplitude lyapunov exponents


for i in range(len(pm.qmins)):
    print('contour locations')
    print(pm.uyminxs[i].x)

amprange = np.arange(0.0, 1.61, 0.05)
numwaves = [1]
nfev = np.zeros(amprange.shape, dtype=np.int32)

for waves in numwaves:
    lyaps = np.zeros(len(amprange))
    lyapstds = np.zeros(len(amprange))
    
    for ind in range(len(amprange)):
        m = amprange[ind]
        print(m, waves)
        
        ampmult = np.ones(numeigs)*m
        ampmult[waves:] = 0
        phaseoffs = np.zeros(numeigs)
        
        for i in range(len(pm.qmins)):
            sol, yclip = pm.generateBreakingSection(ampmult, phaseoffs, pm.qmins[i], resampling=True)
            lyap, lyapstd = pm.findLyapunov(sol, yclip, resampling=True)
            
            if lyap >= lyaps[ind]:
                lyaps[ind] = lyap
                lyapstds[ind] = lyapstd
                
            print('nfev:', sol.nfev)
            
        #lyaps[ind] = lyap
        #lyapstds[ind] = lyapstd
        
        
        
        print('-----')
        
    np.savez('lyapunovs/lyaps_multicontour_{}modes.npz'.format(waves), amprange=amprange, lyaps=lyaps, lyapstds=lyapstds, nfev=nfev)


# %% Phase lyapunov exponents

"""
amprange = np.arange(1.0, 1.16, 0.05)
phrange = np.linspace(-np.pi, np.pi, endpoint=False, num=24)

ampall, phall = np.meshgrid(amprange, phrange)
numall = len(ampall.ravel())

lyaps = np.zeros(numall)
lyapstds = np.zeros(numall)

for ind in range(numall):
    m = ampall.ravel()[ind]
    ph = phall.ravel()[ind]
    print(m, ph)
    
    ampmult = np.ones(numeigs)*m
    phaseoffs = np.zeros(numeigs)
    phaseoffs[0] = ph
    
    for i in range(len(pm.qmins)):
        sol, yclip = pm.generateBreakingSection(ampmult, phaseoffs, pm.qmins[i], resampling=True)
        lyap, lyapstd = pm.findLyapunov(sol, yclip, resampling=True)
        
        if lyap >= lyaps[ind]:
            lyaps[ind] = lyap
            lyapstds[ind] = lyapstd
    
    #sol, yclip = pm.generateBreakingSection(ampmult, phaseoffs, pm.qmin)
    #lyap, lyapstd = pm.findLyapunov(sol, yclip)
    
    #lyaps[ind] = lyap
    #lyapstds[ind] = lyapstd
    
    print('-----')

lyaps = np.reshape(lyaps, ampall.shape)
lyapstds = np.reshape(lyapstds, ampall.shape)

np.savez('lyapunovs/lyaps_multicontour_allphases.npz', amprange=amprange, phrange=phrange, lyaps=lyaps, lyapstds=lyapstds)
"""

# %% Long time lyapunov exponents

"""
timedata = np.load('./poincare_input/eigencomponent_longtimedata.npz')

ampdevs = timedata['ampdevs']
phasedevs = timedata['phasedevs']

t = np.linspace(2400, 3600, num=13, endpoint=True)
lyaps = np.zeros(shape=(3,len(t)))
lyapstds = np.zeros(shape=(3,len(t)))

for ind in range(len(t)):
    print(t[ind])
    
    for i in range(len(pm.qmins)):
        sol, yclip = pm.generateBreakingSection(ampdevs[:,ind], phasedevs[:,ind], pm.qmins[i], resampling=True)
        lyap, lyapstd = pm.findLyapunov(sol, yclip, resampling=True)
        
        lyaps[i,ind] = lyap
        lyapstds[i,ind] = lyapstd
    
    #sol, yclip = pm.generateBreakingSection(ampdevs[:,ind], phasedevs[:,ind], pm.qmin, sections=20, resampling=True)
    #lyap, lyapstd = pm.findLyapunov(sol, yclip, resampling=True)
    
    #lyaps[ind] = lyap
    #lyapstds[ind] = lyapstd
    
    print('-----')
    
np.savez('lyapunovs/lyaps_multicontour_longtimedependent_allmodes.npz', lyaps=lyaps, lyapstds=lyapstds)
"""

# %% Poincare sections via amplitude of waves

"""
amprange = ['000']
#amprange = ['100', '110']

for i in range(len(amprange)):
    m = float(amprange[i])/100.0
    print(m)
    pm.generateFullSection(np.ones(numeigs)*m, np.zeros(numeigs), 'sections/section_amp{}.npz'.format(amprange[i]))
"""

"""
ampmult = np.ones(numeigs)
amps = pm.data['amps']
ampmult[0] = amps[7]/amps[0]
ampmult[7] = amps[0]/amps[7]

print('switched')
pm.generateFullSection(ampmult, np.zeros(numeigs), 'sections/section_switched.npz')
"""