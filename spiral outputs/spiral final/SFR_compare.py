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

def natural_key(path):
    """Sort filenames numerically (e.g., _2.h5 before _10.h5)."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", path)]

def extract_sfr_timeseries(pattern):
    files = sorted(glob.glob(pattern), key=natural_key)
    times = []
    sfr_total = []

    for f in files:
        sim = io_pyro.read(f)
        myg = sim.cc_data.grid
        dens = sim.cc_data.get_var("density")

        sigma_gas = dens[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] * Sigma0_Msun_pc2
        cell_area_kpc2 = (myg.dx * L0_kpc) * (myg.dy * L0_kpc)

        sigma_sfr = np.where(sigma_gas > Sigma_thresh, A_KS * (sigma_gas ** N_KS), 0.0)
        total_sfr = np.sum(sigma_sfr) * cell_area_kpc2

        t_Myr = sim.cc_data.t * (L0_kpc * 3.086e16 / 220.0) / 3.154e13
        times.append(t_Myr)
        sfr_total.append(total_sfr)

    return np.array(times), np.array(sfr_total)

t_sn, sfr_sn = extract_sfr_timeseries("spiral_*.h5")
t_eeos, sfr_eeos = extract_sfr_timeseries("eeos_*.h5")

plt.figure(figsize=(8, 5))
plt.plot(t_sn, sfr_sn, label="Explicit SN + Cooling", linewidth=2, color="#1f77b4")
plt.plot(t_eeos, sfr_eeos, label=r"Effective EOS ($\gamma_{\mathrm{eff}} = 1.35$)",
         linewidth=2, linestyle="--", color="#d62728")

plt.xlabel("Time [Myr]", fontsize=12)
plt.ylabel(r"Total SFR [$M_\odot\ \mathrm{yr}^{-1}$]", fontsize=12)
plt.title("Spiral Galaxy SFR: Explicit Feedback vs. Effective EOS", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("sfr_comparison.png", dpi=200)
plt.show()