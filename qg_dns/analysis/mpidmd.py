from mpi4py import MPI
import numpy as np
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# %% Set up grids, used for certain calculations
nx = 2048

k1 = np.fft.fftfreq(nx, d=1.0/nx)
k2 = np.fft.rfftfreq(nx, d=1.0/nx)

k2g, k1g = np.meshgrid(k2,k1)
lap = k2g**2 + k1g**2
#kd = (k2g>0)*(14.0**2)
kd = 0.0
k2kd = lap+kd
invlap = np.zeros(lap.shape)
invlap[k2kd>0] = -1.0 / k2kd[k2kd>0]
sqk = np.sqrt(-invlap)

# Whether to use energy norm or enstrophy norm
energyNorm = True
compute_pods = True
compute_dmd_modes = False

# Total number of snapshots to use in the POD (need +1 for POD)
totalsnaps = 256
snaps_per_rank = totalsnaps//size
times = np.empty(totalsnaps)
# Note the snapshots are stored row-major order, so it's natural to have snapshots as row vectors
snapshots_all = None # [X* ; x_{m+1}]
# Keep the matrix of snapshots as a slice, to simplify memory management
snapshots = None # X*
if rank == 0:
    snapshots_all = np.empty((totalsnaps+1, nx*nx))
    snapshots = snapshots_all[:-1,:]

if rank == 0:
    print("Loading Data, {} snaps per core".format(snaps_per_rank))

# %% Load data from file
snaps_per_file = 16
if totalsnaps%snaps_per_file == 0:
    nfiles = totalsnaps // snaps_per_file + 1
else:
    nfiles = totalsnaps // snaps_per_file
file_offset = 1 # Normally zero

# Set up proper loading of files
startindex = (rank*snaps_per_rank) % snaps_per_file
snapfile = rank*snaps_per_rank//snaps_per_file

# Check whether or not we need to load multiple files
if startindex+snaps_per_rank > snaps_per_file:
    loadmulti = True
    inds1 = range(startindex, snaps_per_file)
    inds2 = range(0, (startindex+snaps_per_rank)%snaps_per_file)

    inds1c = (startindex, snaps_per_file)
    inds2c = (0, (startindex+snaps_per_rank)%snaps_per_file)
else:
    loadmulti = False
    inds1 = range(startindex, startindex+snaps_per_rank)
    inds1c = (startindex, startindex+snaps_per_rank)

snapshots_chunk = np.empty((snaps_per_rank, nx*nx))
times_chunk = np.empty(snaps_per_rank)

# Load snapshots from files
with h5py.File('../snapshots/snapshots_s{}.h5'.format(snapfile+1+file_offset), mode='r') as simdata:
    qall = simdata['tasks/q']
    # Parallelize the loading of data and converting voriticity into action
    for index in inds1:
        if rank == 0:
            print(index)
        q = qall[index,:,:]
        if energyNorm:
            qfft = np.fft.rfft2(q)
            snapshots_chunk[index-startindex,:] = np.ravel(np.fft.irfft2(sqk * qfft))
        else:
            snapshots_chunk[index-startindex,:] = np.ravel(q*1.0)
    times_chunk[:len(inds1)] = simdata['scales/sim_time'][inds1c[0]:inds1c[1]]

if loadmulti:
    with h5py.File('../snapshots/snapshots_s{}.h5'.format(snapfile+2+file_offset), mode='r') as simdata:
        qall = simdata['tasks/q']
        # Parallelize the loading of data and converting voriticity into action
        for index in inds2:
            if rank == 0:
                print(index)
            q = qall[index,:,:]
            if energyNorm:
                qfft = np.fft.rfft2(q)
                snapshots_chunk[index+snaps_per_file-startindex,:] = np.ravel(np.fft.irfft2(sqk * qfft))
            else:
                snapshots_chunk[index+snaps_per_file-startindex,:] = np.ravel(q*1.0)
        times_chunk[len(inds1):] = simdata['scales/sim_time'][inds2c[0]:inds2c[1]]


#  Load the extra snapshot used for computation of DMD
snapvec_extra = None
if rank == 0:
    extrasnapfile = nfiles
    with h5py.File('../snapshots/snapshots_s{}.h5'.format(extrasnapfile+file_offset), mode='r') as simdata:
        qall = simdata['tasks/q']
        q = qall[totalsnaps%snaps_per_file,:,:]
        # Place the extra snapshot at the end
        snapvec_extra = snapshots_all[-1,:]
        if energyNorm:
            qfft = np.fft.rfft2(q)
            snapvec_extra[:] = np.ravel(np.fft.irfft2(sqk * qfft))
        else:
            snapvec_extra[:] = np.ravel(q*1.0)


# Collect all the times and the snapshots together
dt = None
comm.Gather(times_chunk, times, root=0)
if rank == 0:
    print(times)
    dt = np.median(np.diff(times))
comm.Gather(snapshots_chunk, snapshots, root=0)


# %% Begin computation of SVD for POD modes
if rank == 0:
    print("Computing correlation matrix")

# Parellelize the computation of the correlation matrix
snapvec = np.empty(nx*nx)
correlmat_part = None
correlmat = None # X* X
correlmat_dmd = None # X* Y

if rank == 0:
    correlmat_part = np.empty(totalsnaps)
    correlmat = np.empty((totalsnaps, totalsnaps))
    correlmat_dmd = np.empty((totalsnaps, totalsnaps))

for i in range(totalsnaps):
    if rank == 0:
        if i%4 == 0:
            print("{}".format(i))
        snapvec = snapshots[i,:]
    comm.Bcast(snapvec, root=0)
    correlmat_chunk = snapshots_chunk @ snapvec
    comm.Gather(correlmat_chunk, correlmat_part, root=0)
    if rank == 0:
        correlmat[i,:] = correlmat_part

# Compute the extra column for X* Y
if rank == 0:
    snapvec = snapvec_extra
comm.Bcast(snapvec, root=0)
correlmat_chunk = snapshots_chunk @ snapvec
comm.Gather(correlmat_chunk, correlmat_part, root=0)
if rank == 0:
    correlmat_dmd[:,:-1] = correlmat[:,1:]
    correlmat_dmd[:,-1] = correlmat_part

# Compute POD time traces and energy content
s = np.empty(totalsnaps)
v = np.empty((totalsnaps, totalsnaps))
sinv_vh = np.empty((totalsnaps, totalsnaps)) # \Sigma^{-1} V*
if rank == 0:
    print("Computing V* and Sigma")
    w, v = np.linalg.eigh(-correlmat)
    s = np.sqrt(-w)

# Compute the POD modes
if rank == 0:
    print("Computing and plotting U")
    sinv_vh = np.diag(1/s) @ v.T

sinv_vh_chunk = np.empty(snaps_per_rank)
uh = None
uh_part = None
if rank == 0:
    uh = np.empty((totalsnaps, nx*nx))


if compute_pods:
    # Parallelize matrix multiplication to get POD modes
    for i in range(totalsnaps):
        if rank == 0:
            if i%4 == 0:
                print("{}".format(i))
            uh_part = uh[i,:]

        comm.Scatter(sinv_vh[i,:], sinv_vh_chunk, root=0)
        uh_chunk = sinv_vh_chunk @ snapshots_chunk
        comm.Reduce(uh_chunk, uh_part, op=MPI.SUM, root=0)

        if rank == 0:
            fig, ax = plt.subplots(2,1, figsize=(9.6, 9.6*1.2),
                                   gridspec_kw={'height_ratios':[10,2]})

            if energyNorm:
                plt.suptitle("Energy POD Mode {}, s={}".format(i, s[i]))
            else:
                plt.suptitle("Enstrophy POD Mode {}, s={}".format(i, s[i]))

            if energyNorm:
                toplot = np.fft.irfft2(-1*sqk*np.fft.rfft2(np.reshape(uh_part, (nx, nx))))
            else:
                toplot = np.reshape(uh_part, (nx, nx))

            im = ax[0].imshow(toplot[::2,::2], origin='lower')
            cb = fig.colorbar(im, ax=ax[0])
            trace = ax[1].plot(times, v[:,i], marker='.')

            plt.draw()
            fig.savefig("podmode{:03d}.png".format(i), dpi=100)
            plt.close(fig)
            if i < 7 or i == 37:
                np.save("raw_podmode{:03d}.npy".format(i), toplot)

# Begin computing DMD modes here
dmdrank = 30
atilde = None
if rank == 0:
    print("Computing DMD mode eigenvalues")
    atilde = sinv_vh[:dmdrank,:] @ correlmat_dmd @ (sinv_vh[:dmdrank,:]).T

    wdmd, vdmd = np.linalg.eig(atilde) # wdmd = diag(lambda), vdmd = [ w1 ... wm ]
    # Sort by POD mode energy contribution
    dmdenergy = np.sum(np.abs(vdmd)**2 * s[:dmdrank,np.newaxis]**2, axis=0)
    energyind = np.argsort(-dmdenergy)
    wdmd = wdmd[energyind]
    vdmd = vdmd[:,energyind]
    #print("Frequency of DMD modes")
    dmdfreqs = np.log(wdmd)/dt
    #print("\n".join(map(str, dmdfreqs)))


# Begin plotting and saving information
if rank == 0:
    print("Saving Other Info")

    # Summary figure
    fig, ax = plt.subplots(1,2, figsize=(9.6*2, 9.6*0.5*2))

    ax[0].plot(range(len(s)), np.cumsum(s**2), marker='.')
    ax[0].set_xlabel('POD Mode Number')
    if energyNorm:
        ax[0].set_ylabel('Cumulative Energy Contribution')
    else:
        ax[0].set_ylabel('Cumulative Enstrophy Contribution')
    ax[1].imshow(np.abs(vdmd))
    ax[1].set_xlabel('DMD Mode Number')
    ax[1].set_ylabel('POD Mode Number')
    plt.draw()
    fig.savefig("overview.png", dpi=100)
    plt.close(fig)

    np.savetxt("logdmdfreqs.txt", wdmd)
    np.savetxt("dmdfreqs.txt", dmdfreqs)
    np.savetxt("podsvals.txt", s)

    np.savez("pod_timetraces.npz", v)
    np.savez("dmd_matrix.npz", atilde)


# Compute and plot DMD modes
eigmult_chunk = np.empty(snaps_per_rank, dtype=np.complex)
eigmult = np.empty(dmdrank, dtype=np.complex)
v_sinv = None
snapshots_shifted = None
eig = np.empty(nx*nx, dtype=np.complex)


if compute_dmd_modes:
    if rank == 0:
        print("Computing and plotting DMD modes")
        v_sinv = sinv_vh[:dmdrank,:].T
        snapshots_shifted = snapshots_all[1:,:]
    comm.Scatter(snapshots_shifted, snapshots_chunk, root=0)

    for i in range(dmdrank):
        if rank == 0:
            if i%4 == 0:
                print("{}".format(i))
            eigmult = v_sinv @ (vdmd[:,i] / wdmd[i])

        comm.Scatter(eigmult, eigmult_chunk, root=0)
        eig_chunk = eigmult_chunk @ snapshots_chunk
        comm.Reduce(eig_chunk, eig, op=MPI.SUM, root=0)

        if rank == 0:
            fig, ax = plt.subplots(1,2, figsize=(9.6*2.0, 9.6*1.0))

            plt.suptitle("DMD Mode {}, freq={}".format(i, dmdfreqs[i]))

            if energyNorm:
                toplot = np.fft.irfft2(-1*sqk*np.fft.rfft2(np.reshape(np.real(eig), (nx, nx))))
            else:
                toplot = np.reshape(np.real(eig), (nx, nx))

            im = ax[0].imshow(toplot[::2,::2], origin='lower')
            cb = fig.colorbar(im, ax=ax[0], orientation='horizontal')

            if energyNorm:
                toplot = np.fft.irfft2(-1*sqk*np.fft.rfft2(np.reshape(np.imag(eig), (nx, nx))))
            else:
                toplot = np.reshape(np.imag(eig), (nx, nx))

            im = ax[1].imshow(toplot[::2,::2], origin='lower')
            cb = fig.colorbar(im, ax=ax[1], orientation='horizontal')

            plt.draw()
            fig.savefig("dmdmode{:03d}.png".format(i), dpi=100)
            plt.close(fig)

