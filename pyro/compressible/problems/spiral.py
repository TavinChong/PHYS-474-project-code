import numpy as np
from pyro.util import msg

DEFAULT_INPUTS = "inputs.spiral"

PROBLEM_PARAMS = {
    "spiral.dens0": 1.0,          # Central surface density
    "spiral.dens_bg": 1e-4,       # Ambient background density
    "spiral.r_disk": 0.25,        # Exponential disk scale length
    "spiral.r_core": 0.05,        # Core softening radius
    "spiral.v0": 1.0,             # Asymptotic circular velocity
    "spiral.c_s": 0.05,           # Sound speed (controls disk temperature)
    "spiral.amp": 0.05,           # Spiral perturbation amplitude
    "spiral.m": 2,                # Number of spiral arms
    "spiral.pitch": 15.0,         # Pitch angle in degrees
    "spiral.omega_p": 1.5,        # Pattern speed of the spiral arms
    "spiral.sfr_thresh": 0.1,     # Density threshold in code units (e.g. 10 M_sun/pc^2)
    "spiral.sfr_coeff": 0.015,    # Dimensionless KS coefficient A_KS in code units
    "spiral.sfr_index": 1.4,      # Kennicutt-Schmidt power-law index N
    "spiral.sn_eff": 0.1,         # Thermal feedback coupling efficiency (0.05 - 0.3)
    "spiral.sn_spec_ener": 1.04e4,# (E_SN * eta_SN) / V0^2 in code units
    "spiral.f_return": 0.2,       # Fraction of stellar mass returned by SN winds
    "spiral.cool_coeff": 5.0,     # Dimensionless cooling normalization Lambda_0
    "spiral.c_s_floor": 0.04,     # Minimum sound speed floor (~10 km/s in code units)
    "spiral.c_s_peak": 0.5,       # Sound speed where cooling curve peaks (~10^5.5 K)
    "spiral.use_eeos": 1,         # 0 = explicit SN + cooling, 1 = effective EOS baseline
    "spiral.gamma_eff": 1.35,     # Effective polytropic index (Springel & Hernquist ~ 4/3)
}

def galactic_accel(x, y, t, rp):
    """Compute 2D gravitational acceleration (gx, gy) from background + spiral potential."""
    v0 = rp.get_param("spiral.v0")
    r_core = rp.get_param("spiral.r_core")
    amp = rp.get_param("spiral.amp")
    m = rp.get_param("spiral.m")
    pitch_rad = np.radians(rp.get_param("spiral.pitch"))
    omega_p = rp.get_param("spiral.omega_p")

    r2 = x**2 + y**2 + r_core**2
    r = np.sqrt(r2)
    theta = np.arctan2(y, x)

    # Gravitational velocity curve
    gx = -(v0**2) * x / r2
    gy = -(v0**2) * y / r2

    if amp > 0.0:
        cot_i = 1.0 / np.tan(pitch_rad)
        r_phys = np.sqrt(x**2 + y**2) + 1e-12
        envelope = amp * (r_phys / (r_phys + r_core))**2 * np.exp(-r_phys / 0.8)
        phase = m * (theta - omega_p * t) + m * cot_i * np.log(r / r_core)

        dPhi_dr = envelope * (-m * cot_i / r) * np.sin(phase)
        dPhi_dtheta = -envelope * m * np.cos(phase)

        gr_1 = -dPhi_dr
        gtheta_1 = -(1.0 / r_phys) * dPhi_dtheta

        cos_t = x / r_phys
        sin_t = y / r_phys
        gx += gr_1 * cos_t - gtheta_1 * sin_t
        gy += gr_1 * sin_t + gtheta_1 * cos_t

    return gx, gy

def init_data(my_data, rp):
    """Initialize the rotating galactic gas disk."""
    msg.bold("initializing the spiral galaxy problem...")

    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")

    gamma = rp.get_param("eos.gamma")
    dens0 = rp.get_param("spiral.dens0")
    dens_bg = rp.get_param("spiral.dens_bg")
    r_disk = rp.get_param("spiral.r_disk")
    r_core = rp.get_param("spiral.r_core")
    v0 = rp.get_param("spiral.v0")
    c_s = rp.get_param("spiral.c_s")

    g = my_data.grid
    x = g.x2d
    y = g.y2d
    r = np.sqrt(x**2 + y**2)

    disk_dens = dens0 * np.exp(-r / r_disk)
    dens[:, :] = dens_bg + disk_dens

    taper = 0.5 * (1.0 - np.tanh((r - 0.75) / 0.1))
    v_circ = (v0 * r / np.sqrt(r**2 + r_core**2)) * taper
    r_safe = np.maximum(r, 1e-12)
    vx = -v_circ * (y / r_safe)
    vy = v_circ * (x / r_safe)

    xmom[:, :] = dens * vx
    ymom[:, :] = dens * vy

    c_s_halo = 0.35
    cs_local = c_s + (c_s_halo - c_s) * (dens_bg / dens)
    p = dens * (cs_local**2) / gamma
    ener[:, :] = p / (gamma - 1.0) + 0.5 * dens * (vx**2 + vy**2)

def source_terms(myg, U, ivars, rp):
    """Compute gravity, star formation, and either SN+cooling or effective EOS."""
    dens = U[:, :, ivars.idens]
    xmom = U[:, :, ivars.ixmom]
    ymom = U[:, :, ivars.iymom]
    ener = U[:, :, ivars.iener]

    gamma = rp.get_param("eos.gamma")
    dens_bg = rp.get_param("spiral.dens_bg")
    c_s_floor = rp.get_param("spiral.c_s_floor")

    dens_floor = 0.1 * dens_bg
    dens_safe = np.maximum(dens, dens_floor)

    vx = xmom / dens_safe
    vy = ymom / dens_safe
    spec_ener = ener / dens_safe

    kin_ener = 0.5 * dens_safe * (vx**2 + vy**2)
    e_int = np.maximum(ener - kin_ener, 0.0)
    p = (gamma - 1.0) * e_int
    cs2 = gamma * p / dens_safe

    # Estimate CFL time scale
    v_mag = np.sqrt(vx**2 + vy**2) + np.sqrt(cs2) + 1e-6
    dt_est = min(myg.dx, myg.dy) / v_mag

    S = myg.scratch_array(nvar=ivars.nvar)

    low_dens_mask = dens < dens_floor
    if np.any(low_dens_mask):
        mass_replenish = np.where(
            low_dens_mask,
            0.5 * (dens_floor - dens) / np.maximum(dt_est, 1e-8),
            0.0,
        )
        # Inject replenished gas
        c_s_halo = 0.35
        e_spec_halo = (c_s_halo**2) / (gamma * (gamma - 1.0))

        S[:, :, ivars.idens] += mass_replenish
        S[:, :, ivars.iener] += mass_replenish * e_spec_halo

    # Gravitational acceleration
    gx, gy = galactic_accel(myg.x2d, myg.y2d, 0.0, rp)

    S[:, :, ivars.ixmom] += dens_safe * gx
    S[:, :, ivars.iymom] += dens_safe * gy
    S[:, :, ivars.iener] += xmom * gx + ymom * gy

    # Star Formation Rate
    thresh = rp.get_param("spiral.sfr_thresh")
    a_ks = rp.get_param("spiral.sfr_coeff")
    n_ks = rp.get_param("spiral.sfr_index")

    effective_thresh = np.maximum(thresh, 10.0 * dens_bg)
    sfr = np.where(dens > effective_thresh, a_ks * (dens ** n_ks), 0.0)
    sfr = np.minimum(sfr, 0.05 * dens)

    # Gas Depletion
    f_ret = rp.get_param("spiral.f_return")
    net_depletion = (1.0 - f_ret) * sfr

    S[:, :, ivars.idens] -= net_depletion
    S[:, :, ivars.ixmom] -= net_depletion * vx
    S[:, :, ivars.iymom] -= net_depletion * vy
    S[:, :, ivars.iener] -= net_depletion * spec_ener

    # Thermodynamics
    use_eeos = rp.get_param("spiral.use_eeos")

    if use_eeos:
        # Effective Equation of State
        gamma_eff = rp.get_param("spiral.gamma_eff")
        r_core = rp.get_param("spiral.r_core")

        r_phys = np.sqrt(myg.x2d**2 + myg.y2d**2)
        core_taper = (r_phys**2) / (r_phys**2 + r_core**2)
        eff_index = 1.0 + (gamma_eff - 1.0) * core_taper

        p_floor = dens_safe * (c_s_floor**2) / gamma
        p_eeos = np.where(
            dens > effective_thresh,
            (effective_thresh * (c_s_floor**2) / gamma) * ((dens / effective_thresh) ** eff_index),
            p_floor,
        )

        cs_max = 0.5
        p_max = dens_safe * (cs_max**2) / gamma
        p_eeos = np.minimum(p_eeos, p_max)
        e_target = p_eeos / (gamma - 1.0)

        relax_coeff = np.where(e_int > e_target, 0.25, 0.05)
        relax_rate = relax_coeff * (e_target - e_int) / np.maximum(dt_est, 1e-8)
        S[:, :, ivars.iener] += relax_rate

    else:
        # Supernova Thermal Energy
        sn_eff = rp.get_param("spiral.sn_eff")
        sn_spec_ener = rp.get_param("spiral.sn_spec_ener")
        sn_heating = sn_eff * sn_spec_ener * sfr

        max_sn_heat = 2.0 * dens_safe
        sn_heating = np.minimum(sn_heating, max_sn_heat)
        S[:, :, ivars.iener] += sn_heating

        # Radiative Cooling
        cool_coeff = rp.get_param("spiral.cool_coeff")
        c_s_peak = rp.get_param("spiral.c_s_peak")

        cs2_floor = c_s_floor**2
        cs2_peak = c_s_peak**2

        tau = cs2 / cs2_peak
        lambda_curve = (tau**0.5) / (1.0 + tau**2)
        raw_cooling = cool_coeff * (dens_safe**2) * lambda_curve

        excess_cs2 = np.maximum(cs2 - cs2_floor, 0.0)
        floor_taper = excess_cs2 / (excess_cs2 + 0.25 * cs2_floor)
        bg_mask = np.where(dens > 5.0 * dens_bg, 1.0, 0.0)

        e_excess = dens_safe * excess_cs2 / (gamma * (gamma - 1.0))
        max_cool_rate = 0.1 * e_excess / np.maximum(dt_est, 1e-8)

        cooling = np.minimum(raw_cooling * floor_taper * bg_mask, max_cool_rate)
        S[:, :, ivars.iener] -= cooling

    return S



def finalize():
    """Print post-simulation instructions."""
    print("""
          spiral galaxy simulation complete.
          plot density with: plotvar.py <output_file.h5> density
          """)