# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:35:45 2021

@author: maple
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

font = {'family' : 'serif',
        'size'   : 6}

mpl.rc('font', **font)

# %% formatting for phase

def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int32(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$%s/%s$'%(latex,den)
            elif num==-1:
                return r'$-%s/%s$'%(latex,den)
            else:
                return r'$%s%s/%s$'%(num,latex,den)
    return _multiple_formatter


# %% Figure generated via Scripts/Eigenfunctions/Trajectories/poincare_section_lyapunov.py

tab20c = mpl.cm.get_cmap('tab20c')

fig, ax = plt.subplots(1, 2, figsize=(3.375, 3.375*0.6), dpi=300)


vardata = np.load('eigencomponent_extradata.npz')

mode0_phasedeviation = vardata['mode0_phasedeviation']
energydeviation = vardata['energydeviation']



#amprange = np.arange(0.6, 1.31, 0.05)

ax[0].axvspan(np.sqrt(np.min(energydeviation)), np.sqrt(np.max(energydeviation)), color=tab20c(7.5/20.0))


j = 0
for i in [2, 3, 6, 9]:
    data = np.load('../lyapunovs/lyaps_multicontour_{}modes.npz'.format(i))
    lyaps = data['lyaps']
    lyapstds = data['lyapstds']
    amprange = data['amprange']
    
    #ax[0].errorbar(amprange, lyaps, yerr=lyapstds, c=tab20c((3.5-j)/20.0), label=str(i))
    ax[0].plot(amprange, lyaps, c=tab20c((3.5-j)/20.0), label=str(i))
    j = j+1

ax[0].text(-0.1, 1.02, r'$\bar{\lambda}$', transform=ax[0].transAxes) 
ax[0].legend(loc='upper left')
ax[0].set_yscale('log')
ax[0].set_xlabel('Mode amplitude')
ax[0].yaxis.set_minor_locator(plt.NullLocator())
ax[0].set_ylim([5e-6, 5e2])
ax[0].set_title('vs. number of modes')

data = np.load('../lyapunovs/lyaps_multicontour_allphases.npz')
amprange = data['amprange']
phrange = data['phrange']
lyaps = data['lyaps']
lyapstds = data['lyapstds']

ax[1].axvspan(np.min(mode0_phasedeviation), np.max(mode0_phasedeviation), color=tab20c(7.5/20.0))
#ax[1].axvline(0.0, ls='--', c=tab20c(4.5/20.0))

for i in range(len(amprange)):
    #ax[1].errorbar(phrange, lyaps[:,i], yerr=lyapstds[:,i], c=tab20c((15.5-i)/20.0), label='{0:.2f}'.format(amprange[i]))
    ax[1].plot(phrange, lyaps[:,i], c=tab20c((15.5-i)/20.0), label='{0:.2f}'.format(amprange[i]))
    #ax[1].text(0.05, lyaps[len(phrange)//2,i]*0.7, '{0:.2f}'.format(amprange[i]))
    
ax[1].text(-0.1, 1.02, r'$\bar{\lambda}$', transform=ax[1].transAxes) 
ax[1].legend(loc='upper center', ncol=2)
ax[1].set_yscale('log')
ax[1].set_ylim(None, 2e1)
ax[1].set_xlabel('Mode 1 phase in $x$')
ax[1].xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
ax[1].yaxis.set_minor_locator(plt.NullLocator())
ax[1].set_title('vs. phase of mode 1')

plt.suptitle('Lyapunov Exponents')

plt.tight_layout(h_pad=1.6)
plt.tight_layout(h_pad=1.6)

plt.savefig('lyapunov_exponents.pdf', dpi=200)
plt.savefig('lyapunov_exponents.png', dpi=300)
