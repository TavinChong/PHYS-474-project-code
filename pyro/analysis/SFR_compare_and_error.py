import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from pyro.util import io_pyro

# Physical unit conversions
L0_kpc = 120.0
Sigma0_Msun_pc2 = 3.72759
A_KS = 2.5e-4
N_KS = 1.4
Sigma_thresh = 0.372759
RHO_REL_ERR = 0.157


def natural_key(path):
    """Sort filenames numerically (e.g., _2.h5 before _10.h5)."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path)]


def compute_total_sfr(sigma_gas, cell_area_kpc2):
    """Compute total SFR from gas surface density using the Kennicutt-Schmidt law."""
    sigma_sfr = np.where(sigma_gas > Sigma_thresh, A_KS * (sigma_gas ** N_KS), 0.0)
    return np.sum(sigma_sfr) * cell_area_kpc2


def extract_sfr_timeseries(pattern, compute_bounds=False):
    files = sorted(glob.glob(pattern), key=natural_key)
    times = []
    sfr_total = []
    sfr_low = []
    sfr_high = []

    for f in files:
        sim = io_pyro.read(f)
        myg = sim.cc_data.grid
        dens = sim.cc_data.get_var("density")

        sigma_gas = dens[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] * Sigma0_Msun_pc2
        cell_area_kpc2 = (myg.dx * L0_kpc) * (myg.dy * L0_kpc)

        total_sfr = compute_total_sfr(sigma_gas, cell_area_kpc2)

        t_Myr = sim.cc_data.t * (L0_kpc * 3.086e16 / 220.0) / 3.154e13
        times.append(t_Myr)
        sfr_total.append(total_sfr)

        if compute_bounds:
            sfr_low.append(compute_total_sfr((1.0 - RHO_REL_ERR) * sigma_gas, cell_area_kpc2))
            sfr_high.append(compute_total_sfr((1.0 + RHO_REL_ERR) * sigma_gas, cell_area_kpc2))

    if compute_bounds:
        return (
            np.array(times),
            np.array(sfr_total),
            np.array(sfr_low),
            np.array(sfr_high),
        )
    return np.array(times), np.array(sfr_total)


t_sn, sfr_sn, sfr_sn_low, sfr_sn_high = extract_sfr_timeseries("spiral_*.h5", compute_bounds=True)
t_eeos, sfr_eeos = extract_sfr_timeseries("eeos_*.h5", compute_bounds=False)

yerr_sn = [sfr_sn - sfr_sn_low, sfr_sn_high - sfr_sn]

plt.figure(figsize=(8, 5))
plt.errorbar(
    t_sn,
    sfr_sn,
    yerr=yerr_sn,
    label=r"Explicit SN + Cooling",
    linewidth=2,
    color="#1f77b4",
    ecolor="#1f77b4",
    elinewidth=1.0,
    capsize=2.5,
    alpha=0.9,
)
plt.plot(
    t_eeos,
    sfr_eeos,
    label=r"Effective EOS ($\gamma_{\mathrm{eff}} = 1.35$)",
    linewidth=2,
    linestyle="--",
    color="#d62728",
)

plt.xlabel("Time [Myr]", fontsize=12)
plt.ylabel(r"Total SFR [$M_\odot\ \mathrm{yr}^{-1}$]", fontsize=12)
plt.title("Spiral Galaxy SFR: Explicit Feedback vs. Effective EOS", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("sfr_comparison_error.png", dpi=200)
plt.show()
