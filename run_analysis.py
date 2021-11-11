# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:40:57 2021

@author: maple
"""

import numpy as np
from poincare_map import PoincareMapper

# %% Set up Poincare Mapper

pm = PoincareMapper('poincare_input/poincare_config_fd_smooth.npz')
numeigs = len(pm.data['kys'])


# %% Generate some sections of breaking waves

#pm.generateBreakingSection(np.ones(numeigs)*0.4, np.zeros(numeigs), pm.qmin, 16, 'sections/breaking_amp040.npz')
#pm.generateBreakingSection(np.ones(numeigs)*1.0, np.zeros(numeigs), pm.qmin, 16, 'sections/breaking_amp100.npz')

# %% Generate amplitude lyapunov exponents

amprange = np.arange(0.0, 1.61, 0.05)
numwaves = [2, 3, 6, 9]
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
        
        sol, yclip = pm.generateBreakingSection(ampmult, phaseoffs, pm.qmin, resampling=True)
        lyap, lyapstd = pm.findLyapunov(sol, yclip, resampling=True)
        
        lyaps[ind] = lyap
        lyapstds[ind] = lyapstd
        
        print('nfev:', sol.nfev)
        
        print('-----')
        
    np.savez('lyapunovs/lyaps_{}modes.npz'.format(waves), amprange=amprange, lyaps=lyaps, lyapstds=lyapstds, nfev=nfev)
