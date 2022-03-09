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
import scipy.signal
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
        self.qbarf = qbarf
        
        self.funcs = (psif, utyf, qtf, uyf)
        self.x = x
        
        self.uyminx = scipy.optimize.minimize_scalar(uyf, bounds=(-np.pi, np.pi), method='bounded')
        self.qmin = qbarf(self.uyminx.x) + 8*(self.uyminx.x)
        
        # Compute array of qmins.
        self.uyminxs = x[scipy.signal.argrelextrema(uy, np.less)]
        self.qmins = [qbarf(uymin) + 8*(uymin) for uymin in self.uyminxs]

    # Poincare section method
    def poincareSection(self, ampmult, phaseoffs, z0, sections, zonalmult=1.0, sectionsamps=1):
        data = self.data
        
        qbar = data['qbar']
        
        uy = data['uy']
        psiv = data['psiv']
        freq = data['freq']
        freqs = freq*data['freqmult']
        phases = data['phases']
        kys = data['kys']
        amps = data['amps']
        dopplerc = data['dopplerc']
        numeigs = len(kys)
        
        nonzero_eigs = [i for i in range(numeigs) if ampmult[i] > 1e-8]
        
        psif, utyf, qtf, uyf = self.funcs
        
        amps_mod = amps * ampmult
        phases_mod = phases + phaseoffs
        
        nparticles = len(z0)//2
    
        def f(t,y):
            dy = np.zeros(nparticles*2)
            
            xpts = np.mod(y[:nparticles]+np.pi, 2*np.pi)-np.pi
            ypts = np.mod(y[nparticles:]+np.pi, 2*np.pi)-np.pi
            
            dy[nparticles:] = uyf(xpts)*zonalmult - dopplerc
            
            for i in nonzero_eigs:
                dy[:nparticles] += utyf[i](xpts)*np.cos(kys[i]*ypts - freqs[i]*t - phases_mod[i])*amps_mod[i]
                dy[nparticles:] += -psif[i](xpts)*np.sin(kys[i]*ypts - freqs[i]*t - phases_mod[i])*kys[i]*amps_mod[i]
            
            #utys = np.array(list(utyf[i](xpts)*np.cos(kys[i]*ypts - freqs[i]*t - phases_mod[i])*amps_mod[i] for i in nonzero_eigs))
            #utxs = np.array(list(-psif[i](xpts)*np.sin(kys[i]*ypts - freqs[i]*t - phases_mod[i])*kys[i]*amps_mod[i] for i in nonzero_eigs))
            
            #dy[:nparticles] = np.sum(utxs, axis=0)
            #dy[nparticles:] = np.sum(utys, axis=0) + uyf(xpts)*zonalmult - dopplerc
            
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
                    break
                
            t_eval = t_eval[:len(yclip)]
            sol = TempSol(t=t_eval, y=ytemp, nfev=nfev)
            
            if filename != None:
                ydict = {'yclip{}'.format(ind) : yclip[ind] for ind in range(len(yclip))}
                ydict2 = {'y{}'.format(ind) : sol.y[ind] for ind in range(len(yclip))}
                np.savez(filename, t=sol.t, ampmult=ampmult, phaseoffs=phaseoffs, qcont=qcont, zonalmult=zonalmult, **ydict, **ydict2)
                
            return sol, yclip
        else:
            
            # Compute Poincare section
            sol, yclip = self.poincareSection(ampmult, phaseoffs, z0, sections, zonalmult=zonalmult)
        
            if filename != None:
                np.savez(filename, t=sol.t, y=sol.y, yclip=yclip, ampmult=ampmult, phaseoffs=phaseoffs, qcont=qcont, zonalmult=zonalmult)
            
            return sol, yclip
        
    def fancySpace(self, ampmult, phaseoffs, zonalmult=1.0, nparticles=193):
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
        
        qbarf = self.qbarf
        
        xsamples = np.linspace(-np.pi, np.pi, num=nparticles, endpoint=False)
        qsamples = qbarf(xsamples) + 8*xsamples
        
        z0 = np.ones(nparticles*2) * 8000.0
        
        for i in range(nparticles):
            qcont = qsamples[i]
            
            if xsamples[i] < -3*np.pi/4:
                contours1 = measure.find_contours(qplot, qcont)
                contours2 = measure.find_contours(qplot, qcont+2*8*np.pi)
                contours = contours1 + contours2
            elif xsamples[i] > 3*np.pi/4:
                contours1 = measure.find_contours(qplot, qcont)
                contours2 = measure.find_contours(qplot, qcont-2*8*np.pi)
                contours = contours1 + contours2
            else:
                contours = measure.find_contours(qplot, qcont)
                
            #print("{} - {} contours".format(qcont, len(contours)))
            for c in contours:
                # Ignore contours which begin and end on the same side
                if np.abs(c[0,1] - c[-1,1]) < 1.0:
                    continue
                
                c2 = (2*np.pi*c/2048) - np.pi
                
                ymin = np.argmin(c2[:,1])
                if c2[ymin,1] < z0[nparticles+i]:
                    z0[i] = c2[ymin,0]
                    z0[nparticles+i] = c2[ymin,1]
                    
        return z0

    def generateFullSection(self, ampmult, phaseoffs, filename=None, zonalmult=1.0, nparticles=193, sections=3109, fancyspacing=False):
        #nparticles = 193
        if fancyspacing:
            z0 = self.fancySpace(ampmult, phaseoffs, zonalmult, nparticles)
        else:
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

