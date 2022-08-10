# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:40:57 2021

@author: maple
"""

import numpy as np
from poincare_map import PoincareMapper
import scipy.optimize

# %% Set up Baseline Poincare Mapper

#pm = PoincareMapper('poincare_input/poincare_config_validation.npz')
#numeigs = len(pm.data['kys'])

#pm.generateFullSection(np.ones(numeigs), np.zeros(numeigs), 'sections/section_validation.npz')



# %% Set up Poincare Mapper

suffix = '_uphavg'
#suffix = ''
case = 2
pm = PoincareMapper('poincare_input/case{}_poincare_config_fd_smooth{}.npz'.format(case, suffix))
numeigs = len(pm.data['kys'])

timedata = np.load('poincare_input/case{}_eigencomponent_timedata{}.npz'.format(case, suffix), 'r')


# %% Generate some sections of breaking waves

"""
pm.generateBreakingSection(np.ones(numeigs)*0.4, np.zeros(numeigs), pm.qmin, 8, 'sections/case{}_breaking_amp040.npz'.format(case), resampling=True)
pm.generateBreakingSection(np.ones(numeigs)*1.0, np.zeros(numeigs), pm.qmin, 8, 'sections/case{}_breaking_amp100.npz'.format(case), resampling=True)
pm.generateBreakingSection(np.ones(numeigs)*1.2, np.zeros(numeigs), pm.qmin, 8, 'sections/case{}_breaking_amp120.npz'.format(case), resampling=True)
"""

timeind = 0 if (case==1) else 192
ampmults = timedata['ampdevs'][:,timeind]
phasedevs = timedata['phasedevs'][:,timeind]

uyf = pm.funcs[3]
qbarf = pm.qbarf

uymaxx = scipy.optimize.minimize_scalar(lambda x: -uyf(x), bounds=(-np.pi, np.pi), method='bounded')
qmax = qbarf(uymaxx.x) + 8*(uymaxx.x)

solmin, yclipmin = pm.generateBreakingSection(ampmults, phasedevs, pm.qmin, 16, 'sections/case{}_breaking_qmin_.npz'.format(case), resampling=True)
lyap, lyapstd = pm.findLyapunov(solmin, yclipmin, resampling=True)
print(lyap)

solmax, yclipmax = pm.generateBreakingSection(ampmults, phasedevs, qmax, 16, 'sections/case{}_breaking_qmax_.npz'.format(case), resampling=True)
lyap, lyapstd = pm.findLyapunov(solmax, yclipmax, resampling=True)
print(lyap)

#pm.generateBreakingSection(np.sqrt(pm.data['rsquared'])*0.4, np.zeros(numeigs), pm.qmin, 8, 'sections/case{}_breaking_amp040.npz'.format(case), resampling=True)
#pm.generateBreakingSection(np.sqrt(pm.data['rsquared'])*1.0, np.zeros(numeigs), pm.qmin, 8, 'sections/case{}_breaking_amp100.npz'.format(case), resampling=True)
#pm.generateBreakingSection(np.sqrt(pm.data['rsquared'])*1.2, np.zeros(numeigs), pm.qmin, 8, 'sections/case{}_breaking_amp120.npz'.format(case), resampling=True)


# %% Lyapunov exponents versus number of modes kept

"""
numwaves = [numeigs]
lyaps = np.zeros(len(numwaves))
lyapstds = np.zeros(len(numwaves))
    
    
for ind in range(len(numwaves)):
    waves = numwaves[ind]
    print(waves)
    
    ampmult = np.ones(numeigs)
    ampmult[waves:] = 0
    phaseoffs = np.zeros(numeigs)
    
    for i in range(len(pm.qmins)):
        sol, yclip = pm.generateBreakingSection(ampmult, phaseoffs, pm.qmins[i], sections=8, resampling=True)
        lyap, lyapstd = pm.findLyapunov(sol, yclip, resampling=True)
        
        if lyap >= lyaps[ind]:
            lyaps[ind] = lyap
            lyapstds[ind] = lyapstd
            
        print('nfev:', sol.nfev)
        
    #lyaps[ind] = lyap
    #lyapstds[ind] = lyapstd
    
        
        
    print('-----')
        
    np.savez('lyapunovs/case{}_lyaps_numwaves{}.npz'.format(case, suffix), numwaves=numwaves, lyaps=lyaps, lyapstds=lyapstds)
"""

# %% Generate amplitude lyapunov exponents

"""
print('contour locations')
for i in range(len(pm.qmins)):
    print(pm.uyminxs[i])

amprange = np.arange(0.0, 1.61, 0.05)
#numwaves = reversed(list(range(2,numeigs+1)))
#numwaves = [24]
if case == 1:
    numwaves = [3, 4, 5, 6, 7, 8]
else:
    numwaves = [13, 14, 15, 16, 17, 18]
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
            sol, yclip = pm.generateBreakingSection(ampmult, phaseoffs, pm.qmins[i], sections=8, resampling=True)
            lyap, lyapstd = pm.findLyapunov(sol, yclip, resampling=True)
            
            if lyap >= lyaps[ind]:
                lyaps[ind] = lyap
                lyapstds[ind] = lyapstd
                
            print('nfev:', sol.nfev)
            
        #lyaps[ind] = lyap
        #lyapstds[ind] = lyapstd
        
        
        
        print('-----')
        
    np.savez('lyapunovs/case{}_lyaps_multicontour_{}modes.npz'.format(case, waves), amprange=amprange, lyaps=lyaps, lyapstds=lyapstds, nfev=nfev)
    #np.savez('lyapunovs/case{}_lyaps_multicontour_5modes.npz'.format(case, waves), amprange=amprange, lyaps=lyaps, lyapstds=lyapstds, nfev=nfev)

"""

# %% Phase lyapunov exponents

"""
amprange = np.arange(0.9, 1.15, 0.05)
phrange = np.linspace(-np.pi, np.pi, endpoint=False, num=24)

ampall, phall = np.meshgrid(amprange, phrange)
numall = len(ampall.ravel())

lyaps = np.zeros(numall)
lyapstds = np.zeros(numall)

for ind in range(numall):
    m = ampall.ravel()[ind]
    ph = phall.ravel()[ind]
    print(m, ph)
    
    ampmult = np.ones(numeigs)
    phaseoffs = np.zeros(numeigs)
    phaseoffs[0] = ph
    ampmult[0] = m
    
    for i in range(len(pm.qmins)):
        sol, yclip = pm.generateBreakingSection(ampmult, phaseoffs, pm.qmins[i], resampling=True, sections=8)
        lyap, lyapstd = pm.findLyapunov(sol, yclip, resampling=True)
        
        if lyap >= lyaps[ind]:
            lyaps[ind] = lyap
            lyapstds[ind] = lyapstd
    
    #sol, yclip = pm.generateBreakingSection(ampmult, phaseoffs, pm.qmin)
    #lyap, lyapstd = pm.findLyapunov(sol, yclip)
    
    #lyaps[ind] = lyap
    #lyapstds[ind] = lyapstd
    
    print('-----')
    
    np.savez('lyapunovs/case{}_lyaps_multicontour_allphases.npz'.format(case), amprange=amprange, phrange=phrange, lyaps=np.reshape(lyaps, ampall.shape), lyapstds=np.reshape(lyapstds, ampall.shape))
"""

# %% Phase Lyapunov exponents, in another way

"""
phrange = np.linspace(-np.pi, np.pi, endpoint=False, num=24)

data = np.load('lyapunovs/case{}_lyaps_multicontour_perwavephase.npz'.format(case))

lyaps = data['lyaps']
lyapstds = data['lyapstds']

for k in range(numeigs):
    for j in range(len(phrange)):
        if lyapstds[j,k] > 0:
            continue
        
        ampmult = np.ones(numeigs)
        phaseoffs = np.zeros(numeigs)
        phaseoffs[k] = phrange[j]
        
        print(k, phrange[j])
        
        for i in range(len(pm.qmins)):
            sol, yclip = pm.generateBreakingSection(ampmult, phaseoffs, pm.qmins[i], resampling=True, sections=8)
            lyap, lyapstd = pm.findLyapunov(sol, yclip, resampling=True)
            
            if lyap >= lyaps[j,k]:
                lyaps[j,k] = lyap
                lyapstds[j,k] = lyapstd
        
        print('-----')
        np.savez('lyapunovs/case{}_lyaps_multicontour_perwavephase.npz'.format(case), phrange=phrange, lyaps=lyaps, lyapstds=lyapstds)
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
amprange = ['100']
#amprange = ['100', '110']

numwaves = numeigs

for i in range(len(amprange)):
    m = float(amprange[i])/100.0
    print('sections/case{}_section_amp{}{}.npz'.format(case,amprange[i], suffix))
    
    ampmult = np.ones(numeigs)*m
    ampmult[numwaves:] = 0
    
    #pm.generateFullSection(np.ones(numeigs)*m, np.zeros(numeigs), 'sections/case{}_section_amp{}{}.npz'.format(case,amprange[i], suffix), nparticles=193, sections=3109, fancyspacing=True)
    pm.generateFullSection(ampmult, np.zeros(numeigs), 'sections/case{}_section_amp{}{}.npz'.format(case,amprange[i], suffix), nparticles=521, sections=521, fancyspacing=True)
"""


"""
ampmult = np.ones(numeigs)
amps = pm.data['amps']
ampmult[0] = amps[7]/amps[0]
ampmult[7] = amps[0]/amps[7]

print('switched')
pm.generateFullSection(ampmult, np.zeros(numeigs), 'sections/section_switched.npz')
"""