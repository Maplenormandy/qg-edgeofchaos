# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:24:25 2021

@author: maple
"""

import numpy as np
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import scipy.stats
from skimage import measure
from collections import namedtuple

# %% Load and prepare data

class PoincareMapper:
    def __init__(self, configFile):
        data = np.load(configFile)
        
        self.data = data
        
        qbar = data['qbar']
        
        uy = data['uy']
        psiv = data['psiv']
        freq = data['freq']
        freqs = freq*data['freqmult']
        phases = data['phases']
        kys = data['kys']
        amps = data['amps']
        numeigs = len(kys)
        
        nx = psiv.shape[1]
        kx = np.fft.rfftfreq(nx, 1.0/nx)
        x = np.linspace(-np.pi,np.pi, num=nx, endpoint=False)
        
        
        utildey = np.fft.irfft(-1j*kx[np.newaxis,:]*np.fft.rfft(psiv, axis=1), axis=1)
        qtilde = np.fft.irfft(-(kx[np.newaxis,:]**2 + kys[:,np.newaxis]**2)*np.fft.rfft(psiv, axis=1), axis=1)
        
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
        
        psif = [circularInterpolant(psiv[i,:]) for i in range(numeigs)]
        utyf = [circularInterpolant(utildey[i,:]) for i in range(numeigs)]
        qtf = [circularInterpolant(qtilde[i,:]) for i in range(numeigs)]
        
        uyfft = np.fft.rfft(uy)
        hilbuy = np.fft.irfft(1j*uyfft)
        hilbuyf = circularInterpolant(hilbuy)
        
        uyf = circularInterpolant(uy)
        
        qbarf = circularInterpolant(qbar)
        
        self.funcs = (psif, utyf, qtf, uyf)
        self.x = x
        
        self.uyminx = scipy.optimize.minimize_scalar(uyf, bounds=(-3.0, 0.0), method='bounded')
        self.qmin = qbarf(self.uyminx.x) + 8*(self.uyminx.x)

    # Poincare section method
    def poincareSection(self, ampmult, phaseoffs, z0, sections, zonalmult=1.0, sectionsamps=1, u0=0.0):
        data = self.data
        
        qbar = data['qbar']
        
        uy = data['uy']
        psiv = data['psiv']
        freq = data['freq']
        freqs = freq*data['freqmult']
        phases = data['phases']
        kys = data['kys']
        amps = data['amps']
        numeigs = len(kys)
        
        psif, utyf, qtf, uyf = self.funcs
        
        amps_mod = amps * ampmult
        phases_mod = phases + phaseoffs
        
        nparticles = len(z0)//2
    
        def f(t,y):
            dy = np.zeros(nparticles*2)
            
            xpts = np.mod(y[:nparticles]+np.pi, 2*np.pi)-np.pi
            ypts = np.mod(y[nparticles:]+np.pi, 2*np.pi)-np.pi
            
            utys = np.array(list(utyf[i](xpts)*np.cos(kys[i]*ypts - (freqs[i] + kys[i]*u0)*t - phases_mod[i])*amps_mod[i] for i in range(numeigs)))
            utxs = np.array(list(-psif[i](xpts)*np.sin(kys[i]*ypts - (freqs[i] + kys[i]*u0)*t - phases_mod[i])*kys[i]*amps_mod[i] for i in range(numeigs)))
            
            dy[:nparticles] = np.sum(utxs, axis=0)
            dy[nparticles:] = np.sum(utys, axis=0) + uyf(xpts)*zonalmult + u0
            
            return dy
        
        tmax = (-2*np.pi/freq)*(sections + .01)
        t_eval = np.arange(0, tmax, step=-2.0*np.pi/freq/sectionsamps)
        
        sol = scipy.integrate.solve_ivp(f, [0, tmax], z0, rtol=1e-8, atol=1e-8, t_eval=t_eval)
        yclip = np.mod(sol.y+np.pi, 2*np.pi)-np.pi
        
        return (sol, yclip)

    # Function for outputting the nonlinearity parameter
    def nonlinearParameter(self, ampmult, phaseoffs, zonalmult=1.0, u0=0.0):
        data = self.data
        
        qbar = data['qbar']
        
        uy = data['uy']
        psiv = data['psiv']
        freq = data['freq']
        freqs = freq*data['freqmult']
        phases = data['phases']
        kys = data['kys']
        amps = data['amps']
        numeigs = len(kys)
        
        psif, utyf, qtf, uyf = self.funcs
        
        amps_mod = amps * ampmult
        phases_mod = phases + phaseoffs
        
        xl = np.linspace(-np.pi, np.pi, num=256, endpoint=False)
        
        xg, yg = np.meshgrid(xl,xl)
        z0 = np.zeros(len(np.ravel(xg))*2)
        nparticles = len(z0)//2
        z0[:nparticles] = np.ravel(xg)
        z0[nparticles:] = np.ravel(yg)
        
        def f(t,y):
            dy = np.zeros(nparticles*2)
            
            xpts = np.mod(y[:nparticles]+np.pi, 2*np.pi)-np.pi
            ypts = np.mod(y[nparticles:]+np.pi, 2*np.pi)-np.pi
            
            utys = np.array(list(utyf[i](xpts)*np.cos(kys[i]*ypts - (freqs[i] + kys[i]*u0)*t - phases_mod[i])*amps_mod[i] for i in range(numeigs)))
            utxs = np.array(list(-psif[i](xpts)*np.sin(kys[i]*ypts - (freqs[i] + kys[i]*u0)*t - phases_mod[i])*kys[i]*amps_mod[i] for i in range(numeigs)))
            
            dy[:nparticles] = np.sum(utxs, axis=0)
            dy[nparticles:] = np.sum(utys, axis=0) + uyf(xpts)*zonalmult + u0
            
            return dy
        
        t_eval = np.linspace(0, -2.0*np.pi/freq, num=64)
        minvel = 0.0
        maxspd = 0.0
        for t in t_eval:
            dy = f(t, z0)
            minvel = min(minvel, np.min(dy[nparticles:]))
            maxspd = max(maxspd, np.max(np.sqrt(dy[nparticles:]**2 + dy[:nparticles]**2)))
    
        return (minvel, maxspd)

    # Contour functions
    def resampleContour(self, z):
        nparticles = z.shape[0]//2
        
        # Cumulative arclength along the contour
        arclength_piece = np.sqrt(np.diff(z[:nparticles])**2 + np.diff(z[nparticles:])**2)
        print('shortest/longest segment:', np.min(arclength_piece), np.max(arclength_piece))
        arclength_param = np.zeros(nparticles)
        arclength_param[1:] = np.cumsum(arclength_piece)
        # Arclength of the last segment between the ends
        arclength_lastbit = np.sqrt((np.mod(z[nparticles-1]-z[0]+np.pi,2*np.pi)-np.pi)**2 + (np.mod(z[-1]-z[nparticles]+np.pi,2*np.pi)-np.pi)**2)
        
        arclength = arclength_param[-1]+arclength_lastbit
        
        pad = 4
        
        arclength_pad = np.zeros(len(arclength_param)+2*pad)
        arclength_pad[pad:-pad] = arclength_param
        arclength_pad[:pad] = arclength_param[-pad:] - arclength
        arclength_pad[-pad:] = arclength_param[:pad] + arclength
        
        zxpad = np.zeros(len(arclength_param)+2*pad)
        zxpad[pad:-pad] = z[:nparticles]
        zxpad[:pad] = z[nparticles-pad:nparticles]
        zxpad[-pad:] = z[:pad]
        
        zxf = scipy.interpolate.interp1d(arclength_pad, zxpad, kind='quadratic')
        
        zypad = np.zeros(len(arclength_param)+2*pad)
        zypad[pad:-pad] = z[nparticles:]
        zypad[:pad] = z[-pad:] - 2*np.pi
        zypad[-pad:] = z[nparticles:nparticles+pad] + 2*np.pi
        
        zyf = scipy.interpolate.interp1d(arclength_pad, zypad, kind='quadratic')
        
        num_newsamples = int(arclength/2/np.pi*2048*1.5 + 0.5)
        resampling = np.linspace(0, arclength, num=num_newsamples, endpoint=False)
        zr = np.zeros(2*len(resampling))
        
        print('resampling from/to', nparticles, len(resampling))
        
        zr[:len(resampling)] = zxf(resampling)
        zr[len(resampling):] = zyf(resampling)
        
        return zr
    
    def generateBreakingSection(self, ampmult, phaseoffs, qcont, sections=16, filename=None, resampling=False, zonalmult=1.0, resamplelimit=65536):
        data = self.data
        
        qbar = data['qbar']
        
        uy = data['uy']
        psiv = data['psiv']
        freq = data['freq']
        freqs = freq*data['freqmult']
        phases = data['phases']
        kys = data['kys']
        amps = data['amps']
        numeigs = len(kys)
        
        psif, utyf, qtf, uyf = self.funcs
        x = self.x
        nx=len(x)
        
        amps_mod = amps * ampmult
        phases_mod = phases + phaseoffs
        
        # Compute desried contour
        qts = np.array(list((qtf[i](x)[:,np.newaxis])*(np.cos(kys[i]*x - phases_mod[i])[np.newaxis,:])*amps_mod[i] for i in range(numeigs)))
        qplot = (qbar*zonalmult + 8*x)[:,np.newaxis] + np.sum(qts, axis=0)
        
        contours = measure.find_contours(qplot, qcont)
        print('num contours:', len(contours))
        maxcontour = np.argmax(list(map(len,contours)))
        short_contour = ((contours[maxcontour][::1,:]) / nx * 2 * np.pi) - np.pi
        
        nparticles = short_contour.shape[0]
        z0 = np.zeros(nparticles*2)
        z0[:nparticles] = short_contour[:,0]
        z0[nparticles:] = short_contour[:,1]
        
        if resampling:
            TempSol = namedtuple('TempSol', ['t','y', 'nfev'])
            
            tmax = (-2*np.pi/freq)*(sections + .01)
            t_eval = np.arange(0, tmax, step=-2.0*np.pi/freq)
            
            z0r = self.resampleContour(z0)
            nfev = 0
            
            ytemp = [z0r]
            
            
            yclip = [np.mod(z0+np.pi, 2*np.pi)-np.pi]
            
            for i in range(sections):
                solr, yclipr = self.poincareSection(ampmult, phaseoffs, z0r, 1, zonalmult=zonalmult)
                nfev = nfev + solr.nfev
                yr = self.resampleContour(solr.y[:,-1])
                ytemp.append(yr)
                yclip.append(np.mod(yr+np.pi, 2*np.pi)-np.pi)
                z0r = yr
                if yr.shape[0]//2 > resamplelimit:
                    sol = TempSol(t=t_eval, y=ytemp, nfev=nfev)
                    return sol, yclip
                
            sol = TempSol(t=t_eval, y=ytemp, nfev=nfev)
            return sol, yclip
        else:
            
            # Compute Poincare section
            sol, yclip = self.poincareSection(ampmult, phaseoffs, z0, sections, zonalmult=zonalmult)
        
            if filename != None:
                np.savez(filename, t=sol.t, y=sol.y, yclip=yclip, ampmult=ampmult, phaseoffs=phaseoffs, qcont=qcont, zonalmult=zonalmult)
            
            return sol, yclip

    def generateFullSection(self, ampmult, phaseoffs, filename=None, zonalmult=1.0, nparticles=193, sections=3109):
        #nparticles = 193
        
        z0 = np.zeros(nparticles*2)
        z0[:nparticles] = np.linspace(-np.pi, np.pi, num=nparticles, endpoint=False)
        
        sol, yclip = self.poincareSection(ampmult, phaseoffs, z0, sections, zonalmult=zonalmult)
        
        if filename != None:
            np.savez(filename, t=sol.t, y=sol.y, yclip=yclip, ampmult=ampmult, phaseoffs=phaseoffs, zonalmult=zonalmult)
        
        return sol, yclip
    
    def followTracers(self, ampmult, phaseoffs, filename=None, zonalmult=1.0, nparticles=193):
        z0 = np.zeros(nparticles*2)
        z0[:nparticles] = np.linspace(-np.pi, np.pi, num=nparticles, endpoint=False)
        
        sol, yclip = self.poincareSection(ampmult, phaseoffs, z0, 1, zonalmult=zonalmult, sectionsamps=256)
        
        if filename != None:
            np.savez(filename, t=sol.t, y=sol.y, yclip=yclip, ampmult=ampmult, phaseoffs=phaseoffs, zonalmult=zonalmult)
        
        return sol, yclip

    # More analysis functions
    def findLyapunov(self, sol, yclip, resampling=False):
        
        if resampling:
            arclength = np.zeros(len(yclip))
            
            for j in range(len(yclip)):
                z = sol.y[j]
                nparticles = z.shape[0]//2
                arclength_lastbit = np.sqrt((np.mod(z[nparticles-1]-z[0]+np.pi,2*np.pi)-np.pi)**2 + (np.mod(z[-1]-z[nparticles]+np.pi,2*np.pi)-np.pi)**2)
                arclength[j] = np.sum(np.sqrt(np.diff(z[:nparticles])**2 + np.diff(z[nparticles:])**2))+arclength_lastbit
        else:
            nparticles = yclip.shape[0]//2
            
            arclength_lastbit = np.sqrt((np.mod(sol.y[nparticles-1,:]-sol.y[0,:]+np.pi,2*np.pi)-np.pi)**2 + (np.mod(sol.y[-1,:]-sol.y[nparticles,:]+np.pi,2*np.pi)-np.pi)**2)
            arclength = np.sum(np.sqrt(np.diff(sol.y[:nparticles,:], axis=0)**2 + np.diff(sol.y[nparticles:,:], axis=0)**2), axis=0) + arclength_lastbit
        
        lam = scipy.stats.mstats.linregress(range(len(arclength)), arclength)
        
        print('initial length', arclength[0])
        print('lyap:',lam.slope,lam.stderr)
        
        return lam.slope, lam.stderr

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
    
    sol, yclip = generateBreakingSection(ampmult, phaseoffs, qmin)
    lyap, lyapstd = findLyapunov(sol, yclip)
    
    lyaps[ind] = lyap
    lyapstds[ind] = lyapstd
    
    print('-----')

lyaps = np.reshape(lyaps, ampall.shape)
lyapstds = np.reshape(lyapstds, ampall.shape)

np.savez('lyaps_allphases_test.npz', amprange=amprange, phrange=phrange, lyaps=lyaps, lyapstds=lyapstds)
"""


# %% Zonal lyapunov exponents

"""
zonalrange = np.arange(0.0, 1.61, 0.05)

lyaps = np.zeros(len(zonalrange))
lyapstds = np.zeros(len(zonalrange))

for ind in range(len(zonalrange)):
    zonal = zonalrange[ind]
    print(zonal)
    
    ampmult = np.ones(numeigs)
    phaseoffs = np.zeros(numeigs)
    
    sol, yclip = generateBreakingSection(ampmult, phaseoffs, qmin, zonalmult=zonal)
    lyap, lyapstd = findLyapunov(sol, yclip)
    
    lyaps[ind] = lyap
    lyapstds[ind] = lyapstd
    
    print('-----')
    
np.savez('lyaps_zonal.npz', zonalrange=zonalrange, lyaps=lyaps, lyapstds=lyapstds)

plt.semilogy(zonalrange, lyaps)
"""


# %% Plot contours

"""
plt.figure()

def plotContour(z, c='C0'):
    stride = 1
    nparticles = len(z)//2
    
    chop = np.abs(np.diff(z[nparticles::stride])) > 1.5*np.pi
    chopargs = np.argwhere(chop)[:,0]+1
    
    if len(chopargs)==0:
        plt.plot(z[nparticles:], z[:nparticles], c=c)
    else:
        plt.plot(z[nparticles:nparticles+chopargs[0]:stride], z[:chopargs[0]:stride], c=c)
        plt.plot(z[nparticles+chopargs[-1]::stride], z[chopargs[-1]:nparticles:stride], c=c)
        
        for i in range(len(chopargs)-1):
            plt.plot(z[nparticles+chopargs[i]:nparticles+chopargs[i+1]:stride], z[chopargs[i]:chopargs[i+1]:stride], c=c)

plotContour(yclip[0], c='C0')
plotContour(yclip[-1], c='C1')

plt.figure()

nparticles = sol0.y.shape[0]//2
arclengths_old = np.sum(np.sqrt(np.diff(sol0.y[:nparticles,:], axis=0)**2 + np.diff(sol0.y[nparticles:,:], axis=0)**2), axis=0)

arclengths_new = np.zeros(len(sol.y))
for i in range(len(sol.y)):
    z = sol.y[i]
    nparticles = z.shape[0]//2
    arclength_lastbit = np.sqrt((np.mod(z[nparticles-1]-z[0]+np.pi,2*np.pi)-np.pi)**2 + (np.mod(z[-1]-z[nparticles]+np.pi,2*np.pi)-np.pi)**2)
    arclengths_new[i] = np.sum(np.sqrt(np.diff(z[:nparticles])**2 + np.diff(z[nparticles:])**2))+arclength_lastbit
    
plt.semilogy(arclengths_old)
plt.semilogy(arclengths_new)
"""


# %% Poincare sections via amplitude of waves

"""
amprange = ['11', '12', '16']

for i in range(len(amprange)):
    m = float(amprange[i])/10.0
    print(m)
    generateFullSection(np.ones(numeigs)*m, np.zeros(numeigs), amprange[i]+'_full.npz')
"""

"""
ampmult = np.ones(numeigs)
ampmult[0] = amps[7]/amps[0]
ampmult[7] = amps[0]/amps[7]

generateFullSection(ampmult, np.zeros(numeigs), 'switched_full.npz')
"""

# %% Poincare sections via amplitude of zonal flows

"""
amprange = ['00']

for i in range(len(amprange)):
    m = float(amprange[i])/10.0
    print(m)
    ampmults = np.zeros(numeigs)
    ampmults[0] = 1.0
    generateFullSection(ampmults, np.zeros(numeigs), 'z'+amprange[i]+'_test.npz', zonalmult=m, sections=512)
"""

# %% Generate time-dependent lyapunov exponents

"""
timedata = np.load('./eigencomponent_longtimedata.npz')

ampdevs = timedata['ampdevs']
phasedevs = timedata['phasedevs']

t = np.linspace(2400, 3600, num=13, endpoint=True)
lyaps = np.zeros(t.shape)
lyapstds = np.zeros(t.shape)

for ind in range(len(t)):
    print(t[ind])
    
    sol, yclip = generateBreakingSection(ampdevs[:,ind], phasedevs[:,ind], qmin, sections=20, resampling=True)
    lyap, lyapstd = findLyapunov(sol, yclip, resampling=True)
    
    lyaps[ind] = lyap
    lyapstds[ind] = lyapstd
    
    print('-----')
    
np.savez('lyaps_longtimedependent_allmodes.npz', lyaps=lyaps, lyapstds=lyapstds)
"""

# %%

#plt.figure()
#plt.semilogy(t, lyaps)
#plt.plot(t, ampdevs[0,:])
#plt.plot(t, ampdevs[1,:])
#plt.plot(t, ampdevs[2,:])

# %%

#followTracers(np.ones(numeigs), np.zeros(numeigs), 'tracers_10.npz')