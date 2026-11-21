import glob
import numpy as np
import matplotlib.pyplot as plt
from pyro.util import io_pyro

# Physical unit conversions (example Milky Way scaling)
L0_kpc = 12.0          # 1 code length = 12 kpc
Sigma0_Msun_pc2 = 100.0 # 1 code density = 100 M_sun / pc^2
A_KS = 2.5e-4          # M_sun / yr / kpc^2
N_KS = 1.4
Sigma_thresh = 10.0    # Threshold in M_sun / pc^2 (0.1 in code units)

files = sorted(glob.glob("spiral_*.h5"))
times = []
sfr_total = []

for f in files:
    sim = io_pyro.read(f)
    myg = sim.cc_data.grid
    dens = sim.cc_data.get_var("density")

    # Convert density and cell area to physical units
    sigma_gas = dens[myg.ilo:myg.ihi+1, myg.jlo:myg.jhi+1] * Sigma0_Msun_pc2
    dx_kpc = myg.dx * L0_kpc
    dy_kpc = myg.dy * L0_kpc
    cell_area_kpc2 = dx_kpc * dy_kpc

    # Apply Kennicutt-Schmidt law above threshold
    sigma_sfr = np.where(sigma_gas > Sigma_thresh, A_KS * (sigma_gas ** N_KS), 0.0)

    # Total SFR in M_sun / yr
    total_sfr = np.sum(sigma_sfr) * cell_area_kpc2

    # Convert code time to Myr (t0 = L0 / V0)
    t_Myr = sim.cc_data.t * (L0_kpc * 3.086e16 / 220.0) / 3.154e13
    times.append(t_Myr)
    sfr_total.append(total_sfr)

for t, sfr in zip(times, sfr_total):
    print(f"t = {t:6.1f} Myr | SFR = {sfr:.3f} M_sun/yr")

# Plot Star Formation Rate vs. Time
plt.figure(figsize=(8, 5))
plt.plot(times, sfr_total, marker="o", linewidth=2, markersize=4, color="#1f77b4")
plt.xlabel("Time [Myr]", fontsize=12)
plt.ylabel(r"Total SFR [$M_\odot\ \mathrm{yr}^{-1}$]", fontsize=12)
plt.title("Star Formation Rate vs. Time", fontsize=13)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("sfr_vs_time.png", dpi=200)
plt.show()