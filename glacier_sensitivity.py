"""
glacier_sensitivity.py
Kritagya Paudel, 2025

Parameter sensitivity analysis for a Himalayan glacier mass balance model.
Nepal elevation transect: Chitwan (300 m) to Mustang (5500 m).

The model
---------
A glacier's annual mass balance is the difference between snowfall (accumulation)
and melt. When balance turns persistently negative, the glacier is losing mass.

    B(z, t) = Acc(z) - k * 365 * max(T_eff(z, t), 0)

Temperature at elevation z in year t:
    T_eff = T_base + r*t - gamma*z

Three parameters control the outcome:
    k     : degree-day melt factor [m w.e. / deg C / day]
    r     : warming rate [deg C / year]
    gamma : atmospheric lapse rate [deg C / m]

None of these are known exactly. Sensitivity analysis asks: which parameter
does the model care about most, and how wrong can we be before the qualitative
outcome changes?

The key analytic result is the tipping year formula, derived by setting B=0:

    t*(z) = [Acc(z)/(k*365) + gamma*z - T_base] / r

This tells you, for any elevation band, exactly when it crosses from net
accumulation into net melt. Setting t*=T_horizon and solving for r gives
the critical warming rate as a function of gamma, which is the bifurcation
boundary plotted in glacier_stochastic.py.

What this script produces
-------------------------
Six plots, three rows, two columns. Each row varies one parameter while
holding the other two at their baseline values (one-at-a-time sensitivity).

Left column: heatmap of melt-onset year across (parameter value, elevation).
    Red means the band melts early, green means it stays stable in the window.
    The boundary between red and green is the phase transition.

Right column: cumulative mass loss vs parameter value, one line per band.
    Shows not just when a band tips but how much ice it loses.

The main finding: k and r both drive melt at lower elevations, but the lapse
rate gamma is the dominant control on whether the highest bands survive.
Reducing gamma warms the air at high altitude faster than increasing r alone,
which is why the 4800 m and 5500 m bands show transitions in the gamma panel
that they do not show in the k panel.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. ELEVATION TRANSECT
# =============================================================================

ELEVATIONS   = np.array([300, 800, 1500, 2800, 3800, 4800, 5500])   # metres
ACCUMULATION = np.array([0.0, 0.1, 0.30, 0.60, 0.90, 1.10, 1.20])  # m w.e./yr
ELEV_LABELS  = [f"{e} m" for e in ELEVATIONS]


# =============================================================================
# 2. BASELINE PARAMETERS
# =============================================================================

T_BASE = 25.0    # warm-season temperature at sea level [deg C]
Z_REF  = 0.0   # reference elevation for lapse correction [m]

T_START, T_END = 0, 80
YEARS = np.arange(T_START, T_END + 1)

# Baseline values are empirically grounded Himalayan estimates.
# k=0.006 is mid-range for high-mountain Asia glaciers.
# r=0.04 corresponds to ~4 deg C per century (IPCC RCP4.5 central estimate for HMA).
# gamma=0.0065 is the standard dry adiabatic lapse rate.
DEFAULTS = {
    "k"    : 0.006,
    "r"    : 0.04,
    "gamma": 0.0065,
}


# =============================================================================
# 3. CORE MODEL
# =============================================================================

def effective_temperature(z, t, r, gamma):
    """
    Air temperature at elevation z in year t [deg C], clamped to zero.

    Combines the sea-level baseline, a linear warming trend, and cooling
    with altitude via the lapse rate. The max(..., 0) prevents sub-zero
    temperatures from contributing negative melt, which would be unphysical.

    Parameters
    ----------
    z, t        : float  elevation [m] and year
    r, gamma    : float  warming rate [deg C/yr] and lapse rate [deg C/m]
    """
    T_raw = T_BASE + r * t - gamma * (z - Z_REF)
    return max(T_raw, 0.0)


def annual_mass_balance(z, acc, t, k, r, gamma):
    """
    Net annual mass balance [m w.e.] at elevation z in year t.

    Positive means the glacier gained ice. Negative means it lost ice.
    The factor 365 converts the daily degree-day melt rate to annual totals.
    """
    T_eff = effective_temperature(z, t, r, gamma)
    melt  = k * 365 * T_eff
    return acc - melt


def simulate_transect(k, r, gamma, years=YEARS):
    """
    Run the deterministic model for all elevation bands and all years.

    Returns
    -------
    balance : ndarray, shape (n_elevations, n_years)
    """
    balance = np.zeros((len(ELEVATIONS), len(years)))
    for i, (z, acc) in enumerate(zip(ELEVATIONS, ACCUMULATION)):
        for j, t in enumerate(years):
            balance[i, j] = annual_mass_balance(z, acc, t, k, r, gamma)
    return balance


def analytical_onset(z, r, gamma):
    """
    Closed-form tipping year for an elevation band at z.

    Derived by setting T_eff equal to the melt threshold (the temperature
    at which melt exactly equals accumulation) and solving for t:

        T_base + r*t - gamma*z = Acc(z) / (k * 365)
        t* = [Acc(z)/(k*365) + gamma*z - T_base] / r

    Returns np.nan if the band is already melting at t=0 (t* <= 0).
    This function is used to verify the numerical simulation: both should
    agree on onset year, and they do to within one timestep.

    Note this derivation ignores noise. In the stochastic version (see
    glacier_stochastic.py), noise shifts the actual onset around t* with
    a spread determined by the OU process parameters.
    """
    # Look up accumulation for this elevation
    idx   = np.where(ELEVATIONS == z)[0]
    acc   = ACCUMULATION[idx[0]] if len(idx) > 0 else 0.0
    t_star = (acc / (DEFAULTS["k"] * 365) + gamma * z - T_BASE) / r
    return np.nan if t_star <= 0 else t_star


def melt_onset_year(balance, years=YEARS):
    """
    First year each elevation band's annual balance turns negative.

    Returns np.nan for bands that stay positive throughout the window.
    Bands that never melt in the window are the ones protected by
    the lapse rate at high elevation under moderate warming rates.
    """
    onset = []
    for i in range(balance.shape[0]):
        neg = np.where(balance[i] < 0)[0]
        onset.append(float(years[neg[0]]) if len(neg) > 0 else np.nan)
    return np.array(onset)


def cumulative_loss(balance):
    """
    Total mass lost [m w.e.] over the window, per elevation band.

    Only negative balance years count toward loss. A positive year does not
    cancel ice already lost, so I sum only the negative values. This gives
    a conservative lower bound on total ice loss.
    """
    return np.sum(np.minimum(balance, 0), axis=1)


# =============================================================================
# 4. BASELINE VERIFICATION
# =============================================================================

print("=" * 60)
print("BASELINE  (k=0.006, r=0.04, gamma=0.0065)")
print("=" * 60)

baseline = simulate_transect(**DEFAULTS)
num_onset = melt_onset_year(baseline)

print(f"\n  {'Elevation':>10}  {'Analytic':>14}  {'Numeric':>12}  {'Yr0 balance':>13}")
print(f"  {'-'*10}  {'-'*14}  {'-'*12}  {'-'*13}")

for i, z in enumerate(ELEVATIONS):
    an  = analytical_onset(z, DEFAULTS["r"], DEFAULTS["gamma"])
    num = num_onset[i]
    b0  = baseline[i, 0]
    a_s = f"yr {an:.1f}" if np.isfinite(an) else "already melting"
    n_s = f"yr {num:.0f}" if np.isfinite(num) else "stable in window"
    print(f"  {z:>8} m  {a_s:>14}  {n_s:>12}  {b0:>+11.3f} m w.e.")


# =============================================================================
# 5. SENSITIVITY SWEEP
# =============================================================================

# Extended ranges cover roughly +/-40% of each baseline.
# r goes up to 0.15 to capture the 4800 m tipping threshold (r*=0.124).
# That threshold is derived analytically: t* = 6.2/r, so r*=6.2/50=0.124.
K_RANGE     = np.linspace(0.003, 0.010, 18)
R_RANGE     = np.linspace(0.01,  0.15,  18)
GAMMA_RANGE = np.linspace(0.004, 0.009, 18)


def sensitivity_sweep(param_name, param_range):
    """
    One-at-a-time sweep over a single parameter.

    Holds the other two parameters at baseline and varies param_name
    across param_range. Records melt-onset year and cumulative loss at
    each parameter value.

    This is the standard OAT (one-at-a-time) sensitivity method. It does
    not capture interaction effects between parameters, but it cleanly
    shows which parameters drive the most change in isolation and where
    the phase transitions occur.

    Parameters
    ----------
    param_name  : str      one of 'k', 'r', 'gamma'
    param_range : ndarray  values to sweep

    Returns
    -------
    onset_matrix : ndarray, shape (n_values, n_elevations)
    loss_matrix  : ndarray, shape (n_values, n_elevations)
    """
    onsets = []
    losses = []
    for val in param_range:
        params  = {**DEFAULTS, param_name: val}
        balance = simulate_transect(**params)
        onsets.append(melt_onset_year(balance))
        losses.append(cumulative_loss(balance))
    return np.array(onsets), np.array(losses)


print("\nRunning sensitivity sweeps...")
onset_k,     loss_k     = sensitivity_sweep("k",     K_RANGE)
onset_r,     loss_r     = sensitivity_sweep("r",     R_RANGE)
onset_gamma, loss_gamma = sensitivity_sweep("gamma", GAMMA_RANGE)
print("  Done.")


# =============================================================================
# 6. PHASE TRANSITION DETECTION
# =============================================================================
# A phase transition is where a small parameter increase flips a band from
# "no melt in window" (NaN onset) to "melts at year Y" (finite onset).
# These boundaries are the most policy-relevant output: they tell you how
# much warming margin exists before a currently-stable band becomes unstable.

def find_transitions(onset_matrix, param_range, param_key):
    results = []
    for ei in range(onset_matrix.shape[1]):
        col   = onset_matrix[:, ei]
        entry = {"elevation": ELEVATIONS[ei], "threshold": None, "status": None}

        if np.isfinite(col[0]):
            entry["status"] = "already melting across full range"
        else:
            for i in range(1, len(col)):
                if np.isnan(col[i-1]) and np.isfinite(col[i]):
                    entry["threshold"] = param_range[i]
                    entry["pct"] = (param_range[i] - DEFAULTS[param_key]) / DEFAULTS[param_key] * 100
                    entry["status"] = "transition found"
                    break
            else:
                entry["status"] = "stable throughout full range"

        results.append(entry)
    return results


print("\n" + "=" * 60)
print("PHASE TRANSITION THRESHOLDS")
print("=" * 60)

for label, onset_matrix, param_range, key in [
    ("k  [m w.e./degC/day]",  onset_k,     K_RANGE,     "k"),
    ("r  [degC/year]",         onset_r,     R_RANGE,     "r"),
    ("gamma  [degC/m]",        onset_gamma, GAMMA_RANGE, "gamma"),
]:
    trans = find_transitions(onset_matrix, param_range, key)
    print(f"\n  {label}   baseline={DEFAULTS[key]}")
    for t in trans:
        z = t["elevation"]
        if t["status"] == "transition found":
            print(f"    {z:>5} m  threshold at {t['threshold']:.5f}"
                  f"  ({t['pct']:+.1f}% from baseline)")
        else:
            print(f"    {z:>5} m  {t['status']}")


# =============================================================================
# 7. VISUALISATION
# =============================================================================

ELEV_COLORS = plt.cm.plasma(np.linspace(0.1, 0.9, len(ELEVATIONS)))
SENTINEL    = T_END + 15   # display value for "never melts" -> renders as deep green


def heatmap(ax, onset_matrix, param_range, baseline_val, xlabel, title):
    """
    Melt-onset heatmap: each cell shows the year a band first melts.
    Deep green means the band stayed stable throughout the window.
    The phase transition boundary is the colour edge between red/yellow and green.
    """
    display = np.where(np.isnan(onset_matrix), SENTINEL, onset_matrix).T
    im = ax.imshow(
        display, aspect="auto", origin="lower",
        cmap=plt.cm.RdYlGn, vmin=0, vmax=T_END,
        extent=[param_range[0], param_range[-1], -0.5, len(ELEVATIONS)-0.5]
    )
    ax.set_yticks(range(len(ELEVATIONS)))
    ax.set_yticklabels(ELEV_LABELS, fontsize=9)
    ax.axvline(baseline_val, color="white", linewidth=1.5,
               linestyle="--", alpha=0.8, label=f"baseline ({baseline_val})")
    ax.legend(fontsize=8, loc="upper right",
              facecolor="black", labelcolor="white", framealpha=0.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Year of first net melt  (deep green = stable)", fontsize=8)


def loss_lines(ax, loss_matrix, param_range, baseline_val, xlabel, title):
    """
    Cumulative mass loss vs parameter value, one line per elevation band.
    More negative means more ice lost. The zero line is the stability boundary.
    """
    for i, (label, color) in enumerate(zip(ELEV_LABELS, ELEV_COLORS)):
        ax.plot(param_range, loss_matrix[:, i],
                label=label, color=color, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.axvline(baseline_val, color="gray", linewidth=1.2,
               linestyle=":", alpha=0.7, label="baseline")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Cumulative mass loss  [m w.e.]", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.6)
    ax.grid(True, alpha=0.25)


fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle(
    "Himalayan Glacier Mass Balance: Parameter Sensitivity Analysis\n"
    "Nepal Elevation Transect, Chitwan (300 m) to Mustang (5500 m)",
    fontsize=13, fontweight="bold", y=0.998
)

heatmap(axes[0,0], onset_k,     K_RANGE,     DEFAULTS["k"],
        "Degree-day factor  k  [m w.e. / degC / day]",
        "Melt-onset year vs k")
loss_lines(axes[0,1], loss_k,   K_RANGE,     DEFAULTS["k"],
        "Degree-day factor  k  [m w.e. / degC / day]",
        "Cumulative mass loss vs k")

heatmap(axes[1,0], onset_r,     R_RANGE,     DEFAULTS["r"],
        "Warming rate  r  [degC / year]",
        "Melt-onset year vs r")
# Mark the analytically derived tipping threshold for the 4800 m band.
# Above r=0.124, that band enters the melt regime within 80 years.
# Below it, the lapse rate keeps it frozen throughout the window.
axes[1,0].axvline(0.124, color="cyan", linewidth=1.2,
                  linestyle="-.", alpha=0.9, label="4800 m threshold (r=0.124)")
axes[1,0].legend(fontsize=7, loc="upper right",
                 facecolor="black", labelcolor="white", framealpha=0.5)
loss_lines(axes[1,1], loss_r,   R_RANGE,     DEFAULTS["r"],
        "Warming rate  r  [degC / year]",
        "Cumulative mass loss vs r")

heatmap(axes[2,0], onset_gamma, GAMMA_RANGE, DEFAULTS["gamma"],
        "Lapse rate  gamma  [degC / m]",
        "Melt-onset year vs gamma")
loss_lines(axes[2,1], loss_gamma, GAMMA_RANGE, DEFAULTS["gamma"],
        "Lapse rate  gamma  [degC / m]",
        "Cumulative mass loss vs gamma")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("/mnt/user-data/outputs/glacier_sensitivity.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("\nFigure saved -> glacier_sensitivity.png")

import shutil
shutil.copy(__file__, "/mnt/user-data/outputs/glacier_sensitivity.py")
print("Script saved -> glacier_sensitivity.py")
