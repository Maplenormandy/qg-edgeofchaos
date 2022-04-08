# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:44:49 2021

@author: maple
"""

import numpy as np
import scipy.linalg

# %%

class EigenvalueSolver:
    def __init__(self, qbar, beta=8.0):
        nx = len(qbar)
        
        # Set up spatial operators
        
        # Coarse realspace identity operator I
        ec = np.eye(nx)
        # FFT * I
        fc = np.fft.rfft(ec, axis=0)
        # Refinement * FFT * I
        rfc = np.zeros((nx*3//4+1, nx), dtype=complex)
        rfc[:nx//2+1,:] = fc*1.5
        # IFFT * Refinement * FFT * I
        refine = np.fft.irfft(rfc, axis=0)
        
        # Fine realspace identity operator I
        er = np.eye(nx*3//2)
        # FFT * I
        fr = np.fft.rfft(er, axis=0)
        # Coarsening * FFT * I
        cfr = np.zeros((nx//2+1, nx*3//2), dtype=complex)
        cfr[:,:] = fr[:nx//2+1,:]/1.5
        # IFFT * Coarsening * FFT * I
        coarsen = np.fft.irfft(cfr, axis=0)
        
        refine[np.abs(refine) < 1e-16] = 0
        coarsen[np.abs(coarsen) < 1e-16] = 0
        
        self.refine = refine
        self.coarsen = coarsen
        
        self.fc = fc
        
        x = np.linspace(-np.pi,np.pi, num=nx, endpoint=False)
        kx = np.fft.rfftfreq(nx, 1.0/nx)
        kxf = np.fft.fftfreq(nx, 1.0/nx)
        
        # WARNING! My sign conventions are a bit screwy here because I started in
        # plasma physics units conventions, but switched halfway to using GFD conventions
        # In particular, uy here "actually" means ux!!
        kxinv = np.zeros(kx.shape, dtype=complex)
        kxinv[kx>0] = 1.0/(-1j*kx[kx>0])
        uy = np.fft.irfft(np.fft.rfft(qbar)*kxinv)
        uypp = np.fft.irfft(-1j*kx*np.fft.rfft(qbar))
        b = -beta + uypp
        
        # Multiplication operators
        mb = coarsen @ np.diag(refine @ b) @ refine
        mu = coarsen @ np.diag(refine @ uy) @ refine
        
        self.mb = mb
        self.mu = mu
        self.kx = kx
        self.kxf = kxf
        self.x = x
        
        self.uy = uy
        self.b = b
        
        self.nx = nx

    def solveEigenfunctions(self, ky, kd=0, damping=False, energyNorm=True):
        kx = self.kx
        kxf = self.kxf
        fc = self.fc
        mb = self.mb
        mu = self.mu
    
        # Green's function for psi to q
        greenf = -fc / (kx**2 + kd**2 + ky**2)[:, np.newaxis]
        green = np.fft.irfft(greenf, axis=0)
        
        # Diffusion operator
        if damping:
            difff = fc * ((kx**2 + ky**2)**2)[:, np.newaxis]
            diff = -1j * np.fft.irfft(difff, axis=0)
        
            lu = mu - mb @ green  + 6.0e-9 * diff / ky - 1j * 1.2e-3 / ky
        
            # Solve for eigenvalues and eigenfunctions
            w, vl, vr = scipy.linalg.eig(lu, left=True)
        else:
            lu = mu - mb @ green # + 1.5e-8 * diff / ky # - 1j * alpha / kd
        
            # Solve for eigenvalues and eigenfunctions
            w, vl, vr = scipy.linalg.eig(np.real(lu), left=True)
            
        wind = np.argsort(np.real(w))
        w = w[wind]
        vr = vr[:,wind]
        vl = vl[:,wind]
        
        # Renormalize left eigenvectors so the matrix is more or less the eigenvector projection matrix
        vl = vl/np.conj(np.sum(np.conj(vl)*vr, axis=0)[np.newaxis,:])
        
        invlap = -1.0 / (kxf**2 + kd**2 + ky**2)
        
        vpsi = np.fft.ifft(invlap[:,np.newaxis]*np.fft.fft(vr, axis=0), axis=0)
        # If energyNorm, then eigenvectors should be normalized to unit(-ish) energy, up to FFT normalization
        if energyNorm:
            venergy = -np.real(np.sum(np.conj(vpsi)*vr, axis=0))
            vr = vr/np.sqrt(venergy[np.newaxis,:])
            vl = vl*np.sqrt(venergy[np.newaxis,:])
            vpsi = vpsi/np.sqrt(venergy[np.newaxis,:])
        
        # w is the phase velocity, vr is q, vl is eta
        return {'w':w, 'vr':vr, 'vl':vl, 'vpsi':vpsi}
    
# %%
        
class EigenvalueSolverFD:
    def __init__(self, qbar):
        nx = len(qbar)
        x = np.linspace(-np.pi,np.pi, num=nx, endpoint=False)
        dx = 2*np.pi/nx
        
        # WARNING! My sign conventions are a bit screwy here because I started in
        # plasma physics units conventions, but switched halfway to using GFD conventions
        # In particular, uy here "actually" means ux!!
        uy = np.cumsum(qbar-np.average(qbar))*dx
        uy = -(uy - np.average(uy))
        
        b = np.zeros(nx)
        b[:-1] = -(8+np.diff(qbar)/dx)
        b[-1] = -(8+(qbar[0]-qbar[-1])/dx)
        
        cent_d2x = (np.diag(np.ones(nx-1), 1)+np.diag(np.ones(nx-1), -1) - 2*np.eye(nx) + np.diag(np.ones(1), -(nx-1))+np.diag(np.ones(1), (nx-1))) / dx**2
        
        self.x = x
        self.uy = uy
        self.b = b
        self.cent_d2x = cent_d2x
        self.nx = nx
        
    """
    Possible norms include 'energy', 'enstrophy', and 'action'
    """
    def solveEigenfunctions(self, ky, kd=0, norm='energy'):
        uy = self.uy
        b = self.b
        
        lap = (self.cent_d2x - np.eye(self.nx)*(ky**2 + kd**2))
        invlap = np.linalg.inv(lap)
        
        l = np.diag(-uy) - np.diag(np.sqrt(-b)) @ invlap @ np.diag(np.sqrt(-b))
        w, vh = scipy.linalg.eigh(-l)
        
        vr = vh * np.sqrt(-b)[:,np.newaxis]
        vl = vh / np.sqrt(-b)[:,np.newaxis]
        vpsi = invlap @ vr
        
        if norm=='energy':
            vnorm = -np.sum(vpsi*vr, axis=0)
        elif norm=='enstrophy':
            vnorm = np.sum(vr*vr, axis=0)
        else:
            vnorm = np.ones(self.nx)
        vr = vr/np.sqrt(vnorm[np.newaxis,:])
        vl = vl*np.sqrt(vnorm[np.newaxis,:])
        vpsi = vpsi/np.sqrt(vnorm[np.newaxis,:])
        
        # w is the phase velocity, vr is q, vl is eta
        return {'w':w, 'vr':vr, 'vl':vl, 'vpsi':vpsi, 'vh':vh}


# %%

class EigenvalueSolverFDNonSym:
    def __init__(self, qbar):
        nx = len(qbar)
        x = np.linspace(-np.pi,np.pi, num=nx, endpoint=False)
        dx = 2*np.pi/nx
        
        # WARNING! My sign conventions are a bit screwy here because I started in
        # plasma physics units conventions, but switched halfway to using GFD conventions
        # In particular, uy here "actually" means ux!!
        uy = np.cumsum(qbar-np.average(qbar))*dx
        uy = -(uy - np.average(uy))
        
        b = np.zeros(nx)
        b[:-1] = -(8+np.diff(qbar)/dx)
        b[-1] = -(8+(qbar[0]-qbar[-1])/dx)
        
        cent_d2x = (np.diag(np.ones(nx-1), 1)+np.diag(np.ones(nx-1), -1) - 2*np.eye(nx) + np.diag(np.ones(1), -(nx-1))+np.diag(np.ones(1), (nx-1))) / dx**2
        
        self.x = x
        self.uy = uy
        self.b = b
        self.cent_d2x = cent_d2x
        self.nx = nx
        
    def solveEigenfunctions(self, ky, kd=0):
        uy = self.uy
        b = self.b
        
        lap = (self.cent_d2x - np.eye(self.nx)*(ky**2 + kd**2))
        invlap = np.linalg.inv(lap)
        
        lu = np.diag(uy) - np.diag(b) @ invlap
        w, vl, vr = scipy.linalg.eig(np.real(lu), left=True)
        
        wind = np.argsort(np.real(w))
        w = w[wind]
        vr = vr[:,wind]
        vl = vl[:,wind]
        
        # Renormalize left eigenvectors so the matrix is more or less the eigenvector projection matrix
        vl = vl/np.conj(np.sum(np.conj(vl)*vr, axis=0)[np.newaxis,:])
        
        vpsi = invlap @ vr
        
        return {'w':w, 'vr':vr, 'vl':vl, 'vpsi':vpsi}
    
# %%
def calcPixelEquivLatitude(q, x, beta=8.0):
	qtotal = q + beta*x[:,np.newaxis]
	
	# Flatten the array and sort to count # of pixels with q less than some value
	qsorted = np.sort(np.ravel(qtotal))
	
	# Find i=ind0 where q_i + 2*pi * beta = q_j, and furthemore the # of pixels with q<q_i and q>q_j are the same
	excessarea = qsorted - np.flip(qsorted) + beta*2*np.pi
	ind0 = np.searchsorted(excessarea, 0)
	
	# For the pixels with q<q_i, add beta*2*pi to them so that they're "at the end"
	qsorted2 = np.sort(np.mod(qsorted-qsorted[ind0],beta*2*np.pi))+qsorted[ind0]
	
	# Now, stepping every qtotal.shape[1] pixels gives another step in area. q_0, the area below q_0 and above q_0+2*pi*beta is equal, so it's "0"
	qbar_equiv = qsorted2[::qtotal.shape[1]] - 8*x
	
	return qbar_equiv