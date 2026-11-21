#!/usr/bin/env python3

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt

import pyro.util.io_pyro as io

mpl.rcParams["text.usetex"] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

# font sizes
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'


def makeplot(plotfile_name, outfile, width, height):
    """ plot the data in a plotfile using the solver's dovis() method in astrophysical units """

    sim = io.read(plotfile_name)
    sim.cc_data.fill_BC_all()

    L0_kpc = 120.0
    V0_kms = 220.0
    Sigma0 = 100.0  # M_sun / pc^2
    sim.cc_data.t = sim.cc_data.t * (L0_kpc * 3.086e16 / V0_kms) / 3.154e13

    # CGS conversion factors
    Sigma0_cgs = Sigma0 * 1.9885e33 / (3.0857e18**2)
    V0_cgs = V0_kms * 1e5
    P0_cgs = Sigma0_cgs * (V0_cgs**2)
    e0_cgs = V0_cgs**2

    myg = sim.cc_data.grid
    myg.xmin *= L0_kpc
    myg.xmax *= L0_kpc
    myg.ymin *= L0_kpc
    myg.ymax *= L0_kpc
    myg.dx *= L0_kpc
    myg.dy *= L0_kpc
    myg.x *= L0_kpc
    myg.y *= L0_kpc
    myg.x2d *= L0_kpc
    myg.y2d *= L0_kpc

    dens = sim.cc_data.get_var("density")
    xmom = sim.cc_data.get_var("x-momentum")
    ymom = sim.cc_data.get_var("y-momentum")
    ener = sim.cc_data.get_var("energy")

    dens[:, :] *= Sigma0
    xmom[:, :] *= Sigma0 * V0_kms
    ymom[:, :] *= Sigma0 * V0_kms
    ener[:, :] *= Sigma0 * (V0_kms**2)

    plt.figure(num=1, figsize=(width, height), dpi=100, facecolor='w')

    sim.dovis()
    fig = plt.gcf()

    img_axes = [ax for ax in fig.axes if len(ax.images) > 0]

    unit_titles = [
        r"density [$M_\odot\ \mathrm{pc}^{-2}$]",
        r"$|\mathbf{v}|$ [$\mathrm{km\ s}^{-1}$]",
        r"pressure [$\mathrm{dyn\ cm}^{-1}$]",
        r"specific internal energy [$\mathrm{erg\ g}^{-1}$]",
    ]
    for ax, title in zip(img_axes, unit_titles):
        ax.set_title(title)
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")

    p_to_cgs = P0_cgs / (Sigma0 * (V0_kms**2))
    e_to_cgs = e0_cgs / (V0_kms**2)  # 1e10 (km/s)^2 -> erg/g

    for idx, scale in [(2, p_to_cgs), (3, e_to_cgs)]:
        if len(img_axes) > idx:
            im = img_axes[idx].images[0]
            im.set_data(im.get_array() * scale)
            vmin, vmax = im.get_clim()
            im.set_clim(vmin * scale, vmax * scale)

    if fig.texts:
        fig.texts[0].set_text(f"t = {sim.cc_data.t:.4f} Myr")

    if outfile.endswith(".pdf"):
        plt.savefig(outfile, bbox_inches="tight")
    else:
        plt.savefig(outfile)
    plt.show()


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-o", type=str, default="plot.png",
                        metavar="plot.png", help="output file name")
    parser.add_argument("-W", type=float, default=8.0,
                        metavar="width", help="width (in inches) of the plot (100 dpi)")
    parser.add_argument("-H", type=float, default=8.0,
                        metavar="height", help="height (in inches) of the plot (100 dpi)")
    parser.add_argument("plotfile", type=str, nargs=1,
                        help="the plotfile you wish to plot")

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    makeplot(args.plotfile[0], args.o, args.W, args.H)
