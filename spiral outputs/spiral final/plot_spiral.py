#!/usr/bin/env python3

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt

import pyro.util.io_pyro as io

mpl.rcParams["text.usetex"] = False
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'


def makeplot(plotfile_name, outfile, width, height):
    """ plot the data in a plotfile using the solver's vis() method """

    sim = io.read(plotfile_name)
    sim.cc_data.fill_BC_all()
    sim.cc_data.t = sim.cc_data.t * (12.0  * 3.086e16 / 220.0) / 3.154e13

    plt.figure(num=1, figsize=(width, height), dpi=100, facecolor='w')

    sim.dovis()
    fig = plt.gcf()
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
    parser.add_argument("-H", type=float, default=4.5,
                        metavar="height", help="height (in inches) of the plot (100 dpi)")
    parser.add_argument("plotfile", type=str, nargs=1,
                        help="the plotfile you wish to plot")

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    makeplot(args.plotfile[0], args.o, args.W, args.H)
