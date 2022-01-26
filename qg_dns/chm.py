"""
Dedalus script for Balanced Hasegawa-Wakatani equations

From Majda PoP 2018

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge_procs` command can
be used to merge distributed analysis sets from parallel runs, and the
`plot_slices.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 4 python3 plot_slices.py snapshots/*.h5

This script can restart the simulation from the last save of the original
output to extend the integration.  This requires that the output files from
the original simulation are merged, and the last is symlinked or copied to
`restart.h5`.

To run the original example and the restart, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ ln -s snapshots/snapshots_s2.h5 restart.h5
    $ mpiexec -n 4 python3 rayleigh_benard.py

The simulations should take a few process-minutes to run.

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.core import operators

import h5py

import logging
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Parameters
Lx, Ly = (2*np.pi, 2*np.pi)
Nx, Ny = (2048, 2048)
Beta = 8.0
Viscosity = 1.5e-8
Friction = 1.2e-3

# Create bases and domain
x_basis = de.Fourier('x', Nx, interval=(-Lx/2, Lx/2), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Set up k grid
nx_global = np.array(list(range(Nx//2)))
ny_global = np.array(list(range(Ny//2))+list(range(-Ny//2+1,0)))

nyg_global, nxg_global = np.meshgrid(ny_global, nx_global)
n2_global = nyg_global**2 + nxg_global**2

ky_global, kx_global = np.meshgrid(2*np.pi*ny_global/Ly, 2*np.pi*nx_global/Lx)
k2_global = kx_global**2+ky_global**2

# Set up correlation function for the random forcing
corr_func = np.logical_and(n2_global >= 14**2, n2_global <= 15**2)*1.0
# Inverse laplacian
invlap = np.zeros(k2_global.shape)
invlap[k2_global>0] = 1.0 / k2_global[k2_global>0]
# Compute (lap^-1 C) (0), which is the second line
invlap_corr_func = invlap * corr_func
invlap_corr_total = np.sum(invlap_corr_func[nxg_global>0]*2) + np.sum(invlap_corr_func[nxg_global==0])
# Adjustment factor to match Bouchet choice -- old choice, incorrect
#corr_func = (corr_func / invlap_corr_total) / (0.5 * Lx * Ly)
# Adjustment factor to match Bouchet choice -- new choice, correct and leads to <v^2/2> = 1
corr_func = (corr_func / invlap_corr_total)

# Set up sampling pattern for the random forcing function
cshape = domain.dist.coeff_layout.local_shape(scales=1)
cslice = domain.dist.coeff_layout.slices(scales=1)
# Have to treat nx=0 case carefully, so split out the nx=0 forced modes
nxg_local = nxg_global[cslice]
nyg_local = nyg_global[cslice]
corr_func_local = corr_func[cslice]
forced_local = corr_func_local > 1e-16
# Check if any modes are forced at all
forced_local_nxp = np.logical_and(forced_local, nxg_local > 0)
num_forced_nxp = np.sum(forced_local_nxp)
if num_forced_nxp > 0:
    forced_where = np.where(forced_local_nxp)
    forcing_amps = np.sqrt(corr_func_local[forced_where])
# In the nx=0 case, split into ny<0 and ny>0 modes
forced_local_nx0 = np.logical_and(forced_local, nxg_local == 0)
num_forced_nx0 = np.sum(forced_local_nx0)//2
if num_forced_nx0 > 0:
    forced_where_nx0p = np.where(np.logical_and(forced_local_nx0, nyg_local > 0))
    forced_where_nx0m = np.where(np.logical_and(forced_local_nx0, nyg_local < 0))
    forcing_amps_nx0 = np.sqrt(corr_func_local[forced_where_nx0p])

# Set up empty array for random force
randomforce = np.zeros(cshape, dtype=np.complex)


rng = np.random.default_rng()

def forcing(deltaT):
    if num_forced_nxp > 0:
        noise_r = rng.standard_normal(num_forced_nxp)
        noise_i = rng.standard_normal(num_forced_nxp)
        randomforce[forced_where] = (noise_r+1j*noise_i)*forcing_amps

    if num_forced_nx0 > 0:
        noise_r = rng.standard_normal(num_forced_nx0)
        noise_i = rng.standard_normal(num_forced_nx0)
        randomforce[forced_where_nx0p] = (noise_r+1j*noise_i)*forcing_amps_nx0
        randomforce[forced_where_nx0m] = (noise_r-1j*noise_i)*forcing_amps_nx0

    return randomforce/np.sqrt(deltaT)

forcing_func = operators.GeneralFunction(domain, 'c', forcing, args=[])

# Set up problem equations
problem = de.IVP(domain, variables=['psi', 'vx', 'vy', 'q'])
problem.parameters['Bt'] = Beta
problem.parameters['Mu'] = Viscosity
problem.parameters['Al'] = Friction
problem.parameters['Sq2Al'] = np.sqrt(2.0*Friction)
problem.parameters['Ly'] = Ly
problem.parameters['Lx'] = Lx
problem.parameters['forcing_func'] = forcing_func
problem.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A))"

problem.add_equation("dt(q) + Mu*Lap(Lap(q)) + Al*q - Bt*dy(psi) = -(vx*dx(q) + vy*dy(q)) + Sq2Al*forcing_func")

problem.add_equation("q - Lap(psi) = 0", condition="(nx!=0) or (ny!=0)")
problem.add_equation("psi = 0", condition="(nx==0) and (ny==0)")
problem.add_equation("vy - dx(psi) = 0")
problem.add_equation("vx + dy(psi) = 0")



# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
logger.info('Solver built')

timestep = 2e-5
max_timestep = 0.2
#snapshotStep = 0.0005
snapshotStep = 50.0


# Initial conditions or restart
if not pathlib.Path('restart.h5').exists():
    q = solver.state['q']
    #ff = forcing(1.0)
    q['c'] = forcing(1.0)*np.sqrt(2.0*Friction) + np.logical_and(nxg_local==4, nyg_local==0)*1.5


    # Timestepping and output
    dt = timestep
    stop_sim_time = 3600.1
    fh_mode = 'overwrite'

else:
    # Restart
    #write, last_dt = solver.load_state('restart.h5', -1)
    logger.info("Loading solver state from: restart.h5")

    with h5py.File('restart.h5', mode='r') as f:
        write = f['scales']['write_number'][-1]
        last_dt = f['scales']['timestep'][-1]
        solver.iteration = solver.inital_iteration = f['scales']['iteration'][-1]

        solver.sim_time = solver.initial_sim_time = f['scales']['sim_time'][-1]
        logger.info("Loading iteration: {}".format(solver.iteration))
        logger.info("Loading write: {}".format(write))
        logger.info("Loading sim time: {}".format(solver.sim_time))
        logger.info("Loading timestep: {}".format(last_dt))

        q = solver.state['q']
        psi = solver.state['psi']
        vx = solver.state['vx']
        vy = solver.state['vy']

        last_q = f['tasks']['q'][-1,:,:]
        # Note: I'm not really sure what the conventions for the DFT used in dedalus are, so I use numpy instead

        np_kx = np.fft.fftfreq(Nx, Lx/Nx/2/np.pi)
        np_ky = np.fft.rfftfreq(Ny, Ly/Ny/2/np.pi)

        np_kxg, np_kyg = np.meshgrid(np_kx, np_ky, indexing='ij')
        np_k2 = np_kxg**2 + np_kyg**2
        invlap_np = np.zeros(np_k2.shape)
        invlap_np[np_k2>0] = -1.0 / np_k2[np_k2>0]

        last_qfft = np.fft.rfft2(last_q)
        last_psifft = invlap_np*last_qfft
        last_vxfft = 1j*np_kyg*last_psifft
        last_vyfft = -1j*np_kxg*last_psifft

        last_psi = np.fft.irfft2(last_psifft)
        last_vx = np.fft.irfft2(last_vxfft)
        last_vy = np.fft.irfft2(last_vyfft)

        gshape = domain.dist.grid_layout.local_shape(scales=1)
        gslice = domain.dist.grid_layout.slices(scales=1)

        q['g'] = last_q[gslice]
        psi['g'] = last_psi[gslice]
        vx['g'] = last_vx[gslice]
        vy['g'] = last_vy[gslice]


    # Timestepping and output
    dt = last_dt
    stop_sim_time = 3600.1
    fh_mode = 'append'

# Integration parameters
solver.stop_sim_time = stop_sim_time

# Analysis
# Note: for some reason, if I don't include this line, the solver refuses to start. I'm not sure why.
fh_fixer = solver.evaluator.add_file_handler('fh_fixer', sim_dt=10000.0, max_writes=10, mode=fh_mode)
fh_fixer.add_system(solver.state)
# This is the actual
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=snapshotStep, max_writes=8, mode=fh_mode)
snapshots.add_task('q', layout='g', name='q')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1.0,
                     max_change=1.5, min_change=0.1, max_dt=max_timestep, threshold=0.05)
CFL.add_velocities(('vx', 'vy'))

# Flow properties
output_cadence = 10

flow = flow_tools.GlobalFlowProperty(solver, cadence=output_cadence)
flow.add_property("vx*vx + vy*vy", name='Energy')
flow.add_property("q*q", name='Enstrophy')

curr_time = time.time()


# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    start_iter = solver.iteration
    curr_iter = solver.iteration

    while solver.proceed:
        if solver.iteration - start_iter > 10:
            dt = CFL.compute_dt()

        forcing_func.args = [dt]
        dt = solver.step(dt)
        if (solver.iteration-2) % output_cadence == 0:
            next_time = time.time()
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Average timestep (ms): %f' % ((next_time-curr_time) * 1000.0 / (solver.iteration - curr_iter)))
            logger.info('Max v^2 = %f' % flow.max('Energy'))
            logger.info('Average v^2 = %f' % flow.volume_average('Energy'))
            logger.info('Max q^2 = %f' % flow.max('Enstrophy'))
            curr_time = next_time
            curr_iter = solver.iteration
            if solver.iteration - start_iter > 100:
                output_cadence = 100
            if not np.isfinite(flow.max('Enstrophy')):
                raise Exception('NaN encountered')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
