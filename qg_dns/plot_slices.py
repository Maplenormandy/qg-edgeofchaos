"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
from os import path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ioff()
#from dedalus.extras import plot_tools


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    #scale = 2.5
    dpi = 75
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    # Set up a new colorscale
    colors = plt.cm.twilight(np.linspace(0,1,32))

    numzones = 3
    colors2 = np.vstack(list([colors for i in range(numzones+2)]))
    qbrange = 8.0*(2*np.pi/numzones)*(numzones+2)
    mymap2 = mpl.colors.LinearSegmentedColormap.from_list('twilight_stacked', colors2)


    # Plot writes
    with h5py.File(filename, mode='r') as simdata:
        # Set up scales for Fourier transform
        x = simdata['scales/x']['1.0'][:]
        y = simdata['scales/y']['1.0'][:]

        nx = len(x)
        ny = len(y)
        kx = np.fft.fftfreq(nx, 1.0/nx)
        ky = np.fft.rfftfreq(ny, 1.0/ny)

        kxg, kyg = np.meshgrid(kx, ky, indexing='ij')
        xg, yg = np.meshgrid(x, y, indexing='ij')

        k2 = kxg**2 + kyg**2
        invlap = np.zeros(k2.shape)
        invlap[k2>0] = -1.0 / k2[k2>0]

        for index in range(start, start+count):
            # Check if plot already exists to not duplicate work
            savename = savename_func(simdata['scales/write_number'][index])
            savepath = output.joinpath(savename)

            if path.exists(str(savepath)):
                continue

            fig, ax = plt.subplots(2,4, figsize=(4.0*9.6, 4.0*3.0),
                                   gridspec_kw={'width_ratios':[29,29,29,9], 'height_ratios':[29,1]})

            # Add time title
            title = title_func(simdata['scales/sim_time'][index])
            plt.suptitle(title)
            plt.tight_layout()

            # Plot data
            q = simdata['tasks/q'][index,:,:]
            qfft = np.fft.rfft2(q)
            psifft = invlap*qfft
            vxfft = 1j*kyg*psifft
            vyfft = -1j*kxg*psifft
            psi = np.fft.irfft2(psifft)
            vx = np.fft.irfft2(vxfft)
            vy = np.fft.irfft2(vyfft)

            vybar = np.average(vy, axis=1)

            psibar = np.average(psi, axis=1)
            psitilde = psi-psibar[:,np.newaxis]
            psimax = np.max(np.abs(psitilde))

            qbar = np.average(q, axis=1)
            qtilde = q-qbar[:,np.newaxis]
            qmax = np.max(np.abs(qtilde))

            # Attempt to align the zonal color with the phase of q
            offset = np.angle(qfft[numzones,0])+np.pi/2.0
            if offset>np.pi:
                offset = offset-np.pi
            qplot = q+8.0*(x[:,np.newaxis]+offset)

            cf = ax[0,1].pcolormesh(yg.T, xg.T, np.flipud(psitilde.T), cmap='viridis', vmin=-psimax, vmax=psimax, shading='auto')
            ax[0,1].set_aspect('equal')
            fig.colorbar(cf, cax=ax[1,1], orientation='horizontal')

            cf = ax[0,2].pcolormesh(yg.T, xg.T, np.flipud(qplot.T), cmap=mymap2, shading='gouraud', vmin=-qbrange/2.0, vmax=qbrange/2.0)
            ax[0,2].set_aspect('equal')
            fig.colorbar(cf, cax=ax[1,2], orientation='horizontal')

            cf = ax[0,0].pcolormesh(yg.T, xg.T, np.flipud(qtilde.T), cmap='viridis', shading='gouraud', vmin=-qmax, vmax=qmax)
            ax[0,0].set_aspect('equal')
            fig.colorbar(cf, cax=ax[1,0], orientation='horizontal')

            ax[0,3].plot(vybar, x)
            ax[0,3].axvline(0.0, ls='--')
            ax[0,3].set_ylim([-np.pi, np.pi])

            # Save figure
            fig.savefig(str(savepath), dpi=dpi)
            #fig.clear()
            plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

