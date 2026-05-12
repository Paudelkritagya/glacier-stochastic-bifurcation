"""
glacier_stochastic.py
Kritagya Paudel, 2025

Himalayan glacier mass balance model with stochastic temperature forcing,
Monte Carlo simulation, bifurcation analysis, and early warning signal detection.

Background
----------
A glacier stays healthy as long as annual snowfall (accumulation) exceeds
annual melt. The difference is called mass balance:

    B(z, t) = Acc(z) - k * 365 * max(T_eff(z, t), 0)

where z is elevation, t is year, k is the degree-day melt factor, and
T_eff is the effective air temperature at that elevation and year.

The twist in this version is that temperature is not a smooth linear trend.
Real interannual climate variability is noisy and mean-reverting, so I model
it with an Ornstein-Uhlenbeck process:

    T_eff(z, t) = T_base + r*t + eta(t) - gamma*z

    d(eta) = -theta * eta * dt + sigma * dW

This is the same SDE used in the Vasicek interest rate model. The noise
reverts to zero with timescale 1/theta, so a warm year is followed by
slightly-above-average temperatures before things normalize, not perpetual
warming. That makes it physically appropriate for annual climate anomalies.

Why bother with stochastic forcing? Three reasons:

1. A single deterministic run gives one tipping year. Running 500 noisy
   simulations gives a full probability distribution of tipping times,
   which is much more useful for risk assessment.

2. Near a tipping point, the system exhibits critical slowing down: recovery
   from perturbations gets slower as the bifurcation approaches. This shows
   up as rising variance and rising lag-1 autocorrelation in the time series,
   both of which can be detected before the actual collapse.

3. The early warning signal math (Scheffer et al. 2009, Dakos et al. 2008)
   is the same framework used in quantitative finance to detect market regime
   changes before they happen. A glacier tipping into melt and a market
   tipping into a trend are both stochastic bifurcations governed by the
   same equations.

Focus elevation: 4800 m
    At baseline warming r=0.04: tipping year t* = 155, well outside window.
    At scenario r=0.12: t* = 6.702/0.12 = 55.9 years. This puts the tipping
    point near the middle of the 80-year window, giving the early warning
    signals enough time to build before collapse and enough post-tip data
    to verify the detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import gaussian_kde

np.random.seed(42)


# =============================================================================
# 1. ELEVATION TRANSECT
# =============================================================================

# Seven elevation bands from the Chitwan valley up to high Mustang.
# Accumulation increases with altitude because high elevations receive
# snow rather than rain, and temperatures are too cold for significant
# melt to offset it.
ELEVATIONS   = np.array([300,  800,  1500,  2800,  3800,  4800,  5500])  # metres
ACCUMULATION = np.array([0.0,  0.1,  0.30,  0.60,  0.90,  1.10,  1.20]) # m w.e./yr
ELEV_LABELS  = [f"{e} m" for e in ELEVATIONS]

# 4800 m is the focus band for the stochastic analysis. It sits comfortably
# above the tipping boundary under current warming but crosses it under the
# high-emissions scenario (r=0.12), which is what makes it interesting.
FOCUS_IDX  = 5
FOCUS_ELEV = ELEVATIONS[FOCUS_IDX]
FOCUS_ACC  = ACCUMULATION[FOCUS_IDX]


# =============================================================================
# 2. PARAMETERS
# =============================================================================

T_BASE    = 25.0   # warm-season surface temperature at sea level [deg C]
Z_REF     = 0.0   # reference elevation for lapse-rate correction [m]
T_HORIZON = 80    # simulation length [years]
YEARS     = np.arange(T_HORIZON + 1)
N_RUNS    = 500   # Monte Carlo ensemble size

K_MELT     = 0.006   # degree-day melt factor [m w.e. / deg C / day]
R_SCENARIO = 0.12    # warming rate for this analysis [deg C / yr]
GAMMA      = 0.0065  # atmospheric lapse rate [deg C / m]

# OU noise parameters. Theta=0.4 gives a memory timescale of about 2.5 years,
# which is consistent with observed Himalayan interannual temperature persistence.
# Sigma=1.2 deg C matches the standard deviation of annual temperature anomalies
# in high-mountain Asia climate records.
THETA = 0.4
SIGMA = 1.2

EWS_WINDOW = 12  # rolling window length for early warning statistics [years]


# =============================================================================
# 3. ORNSTEIN-UHLENBECK NOISE
# =============================================================================

def ou_process(n_steps, theta, sigma, n_runs):
    """
    Generate an ensemble of Ornstein-Uhlenbeck noise trajectories.

    The OU process is a continuous-time SDE:
        d(eta) = -theta * eta * dt + sigma * dW

    The key property is mean reversion. If eta is large and positive,
    the drift term -theta*eta pulls it back toward zero. The rate of
    reversion is controlled by theta: larger theta means faster reversion
    and shorter memory (1/theta years).

    I use the exact discrete update derived by Gillespie (1996) rather
    than an Euler approximation. For a timestep dt=1 year, this gives:

        eta(t+1) = exp(-theta) * eta(t)
                 + sigma * sqrt((1 - exp(-2*theta)) / (2*theta)) * eps
        eps ~ N(0, 1)

    This is exact because it solves the SDE analytically over each interval,
    not just approximates the derivative. At the scales we care about (years,
    not days), this matters for getting the variance structure right.

    Parameters
    ----------
    n_steps : int    number of annual timesteps to simulate
    theta   : float  mean-reversion speed [1/yr]
    sigma   : float  noise amplitude [deg C]
    n_runs  : int    number of independent climate realisations

    Returns
    -------
    eta : ndarray, shape (n_runs, n_steps+1)
        Each row is one independent noise trajectory.
    """
    decay     = np.exp(-theta)
    noise_std = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta)) / (2.0 * theta))

    eta = np.zeros((n_runs, n_steps + 1))
    for t in range(n_steps):
        eps         = np.random.standard_normal(n_runs)
        eta[:, t+1] = decay * eta[:, t] + noise_std * eps

    return eta


# =============================================================================
# 4. STOCHASTIC MASS BALANCE SIMULATION
# =============================================================================

def simulate_stochastic(z, acc, k, r, gamma, eta_ensemble):
    """
    Compute annual mass balance for one elevation band across all Monte Carlo runs.

    For each run i and year t:
        T_eff = T_BASE + r*t + eta[i,t] - gamma*(z - Z_REF)
        melt  = k * 365 * max(T_eff, 0)
        B[i,t] = acc - melt

    The max(..., 0) on T_eff is physically necessary: sub-zero temperatures
    produce no melt. Without this, a cold year would appear to "add" mass
    through the melt term, which is wrong.

    Parameters
    ----------
    z            : float    elevation [m]
    acc          : float    annual snowfall accumulation [m w.e.]
    k, r, gamma  : float    melt factor, warming rate, lapse rate
    eta_ensemble : ndarray  shape (n_runs, n_years), OU noise trajectories

    Returns
    -------
    balance : ndarray, shape (n_runs, n_years)
        Annual mass balance for each run and year.
    """
    n_runs  = eta_ensemble.shape[0]
    balance = np.zeros((n_runs, len(YEARS)))

    for j, t in enumerate(YEARS):
        T_eff         = T_BASE + r*t + eta_ensemble[:, j] - gamma*(z - Z_REF)
        T_eff         = np.maximum(T_eff, 0.0)
        balance[:, j] = acc - k * 365 * T_eff

    return balance


def melt_onset_ensemble(balance, consec=5):
    """
    Find the tipping year for each Monte Carlo run.

    A run has tipped when its mass balance stays negative for `consec`
    consecutive years. Requiring consecutive years filters out brief
    noise-driven dips that don't represent genuine regime change. A single
    bad year followed by recovery is not a tipping event; five consecutive
    bad years almost certainly is.

    Returns np.nan for runs that never tip within the simulation window.

    Parameters
    ----------
    balance : ndarray, shape (n_runs, n_years)
    consec  : int  minimum consecutive negative years to count as tipping

    Returns
    -------
    onset : ndarray, shape (n_runs,)  tipping year per run, or np.nan
    """
    n_runs = balance.shape[0]
    onset  = np.full(n_runs, np.nan)

    for i in range(n_runs):
        for j in range(len(YEARS) - consec):
            if np.all(balance[i, j:j+consec] < 0):
                onset[i] = YEARS[j]
                break

    return onset


# =============================================================================
# 5. EARLY WARNING SIGNALS
# =============================================================================

def rolling_variance(series, window):
    """
    Compute rolling variance with a sliding window.

    Near a tipping point, the system's restoring force weakens. Random shocks
    push the mass balance further from equilibrium before it recovers, so the
    variance of year-to-year fluctuations increases. By the fluctuation-
    dissipation theorem, Var(B) ~ sigma^2 / (2 * |lambda|), where lambda is
    the dominant eigenvalue. As lambda approaches 1 at the bifurcation,
    variance diverges. This makes rising variance a leading indicator of
    approaching collapse.

    Parameters
    ----------
    series : ndarray  annual mass balance time series
    window : int      rolling window length [years]

    Returns
    -------
    out : ndarray  rolling variance, NaN for the first `window` entries
    """
    n   = len(series)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = np.var(series[i-window:i], ddof=1)
    return out


def rolling_ac1(series, window):
    """
    Compute rolling lag-1 autocorrelation with a sliding window.

    AC1 measures how strongly each year's mass balance predicts the next
    year's. In a stable system, a bad year is quickly followed by recovery,
    so AC1 is low. Near a tipping point, recovery slows down: a bad year
    leaves the system below equilibrium for longer, making the next year
    also likely to be bad. AC1 rises toward 1 as the bifurcation approaches,
    because the linearised dynamics satisfy: AC1 = exp(lambda) -> 1 as
    lambda -> 0. Both variance and AC1 diverge at the same point.

    Parameters
    ----------
    series : ndarray  annual mass balance time series
    window : int      rolling window length [years]

    Returns
    -------
    out : ndarray  rolling lag-1 autocorrelation, NaN for first `window` entries
    """
    n   = len(series)
    out = np.full(n, np.nan)

    for i in range(window, n):
        chunk = series[i-window:i]
        if np.std(chunk) < 1e-10:
            continue
        # Pearson correlation between the chunk and itself shifted by one year
        out[i] = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]

    return out


def kendall_tau(signal):
    """
    Test whether the signal has a statistically significant rising trend.

    Kendall's tau is a rank correlation between the signal values and their
    time indices. tau=+1 means perfectly monotone increase, tau=0 means no
    trend. I use this rather than Pearson correlation because it makes no
    assumptions about linearity or normality, which is appropriate for
    variance and autocorrelation signals that can behave nonlinearly near
    a bifurcation.

    This is the standard test from Dakos et al. (2008, PNAS) for evaluating
    early warning signals in tipping-point systems.

    Only computed on the non-NaN portion (after the rolling window fills).

    Parameters
    ----------
    signal : ndarray  time series of the EWS statistic (variance or AC1)

    Returns
    -------
    tau : float  Kendall rank correlation
    p   : float  two-sided p-value
    """
    valid = ~np.isnan(signal)
    if valid.sum() < 10:
        return np.nan, np.nan
    idx       = np.where(valid)[0]
    tau, p    = stats.kendalltau(idx, signal[valid])
    return tau, p


# =============================================================================
# 6. ANALYTICAL BIFURCATION BOUNDARY
# =============================================================================

def critical_r(gamma_arr, z, acc, k, horizon):
    """
    Compute the critical warming rate for elevation z as a function of gamma.

    Setting mass balance to zero and solving for the tipping year gives:
        t*(z) = [acc/(k*365) + gamma*z - T_BASE] / r

    Setting t* = horizon and solving for r gives the critical curve:
        r_crit(gamma, z) = [acc/(k*365) + gamma*z - T_BASE] / horizon

    For r below this curve the glacier survives the simulation window.
    Above it the glacier collapses. The boundary is linear in gamma for
    fixed z, so each elevation band produces a straight line in (r, gamma)
    space. The derivation assumes no noise (deterministic skeleton); noise
    shifts the effective boundary slightly but this gives the right structure.

    Parameters
    ----------
    gamma_arr : ndarray  lapse rate values to evaluate [deg C/m]
    z         : float    elevation [m]
    acc       : float    annual accumulation [m w.e./yr]
    k         : float    degree-day melt factor
    horizon   : int      simulation window [years]

    Returns
    -------
    r_crit : ndarray  critical warming rate, NaN where physically undefined
    """
    thresh = acc / (k * 365)
    r_c    = (thresh + gamma_arr * z - T_BASE) / horizon
    return np.where(r_c > 0, r_c, np.nan)


# =============================================================================
# 7. RUN THE SIMULATION
# =============================================================================

print("=" * 60)
print("HIMALAYAN GLACIER STOCHASTIC BIFURCATION MODEL")
print("=" * 60)

# Analytical tipping year at the scenario warming rate (no noise)
thresh_4800 = FOCUS_ACC / (K_MELT * 365)
t_star      = (thresh_4800 + GAMMA * FOCUS_ELEV - T_BASE) / R_SCENARIO

print(f"\nFocus band          : {FOCUS_ELEV} m")
print(f"Warming rate        : r = {R_SCENARIO} deg C/yr")
print(f"Analytical tip year : t* = {t_star:.1f}  (deterministic, no noise)")
print(f"OU noise            : theta={THETA}, sigma={SIGMA} deg C")
print(f"Ensemble size       : N = {N_RUNS} runs\n")

print("Generating OU noise ensemble...")
eta = ou_process(T_HORIZON, THETA, SIGMA, N_RUNS)

print("Simulating stochastic mass balance...")
bal = simulate_stochastic(FOCUS_ELEV, FOCUS_ACC, K_MELT, R_SCENARIO, GAMMA, eta)

print("Finding tipping years...")
onset  = melt_onset_ensemble(bal, consec=5)
tipped = onset[~np.isnan(onset)]
n_tip  = len(tipped)
n_surv = N_RUNS - n_tip

print(f"  Tipped  : {n_tip}/{N_RUNS}  ({100*n_tip/N_RUNS:.1f}%)")
if n_tip > 0:
    print(f"  P10 / Median / P90 : "
          f"{np.percentile(tipped,10):.0f} / "
          f"{np.median(tipped):.0f} / "
          f"{np.percentile(tipped,90):.0f} years")

mean_bal = np.mean(bal, axis=0)
p10_bal  = np.percentile(bal, 10, axis=0)
p90_bal  = np.percentile(bal, 90, axis=0)

# Compute EWS for every run then average across the ensemble.
# Averaging first suppresses run-to-run noise in the rolling statistics,
# making the underlying trend much cleaner.
print("\nComputing early warning signals...")
var_runs = np.array([rolling_variance(bal[i], EWS_WINDOW) for i in range(N_RUNS)])
ac1_runs = np.array([rolling_ac1(bal[i],      EWS_WINDOW) for i in range(N_RUNS)])

mean_var = np.nanmean(var_runs, axis=0)
mean_ac1 = np.nanmean(ac1_runs, axis=0)

# Restrict Kendall tau to the pre-tipping window to avoid contamination
# from post-collapse dynamics, which would inflate both statistics trivially.
pre_tip  = YEARS < (t_star - 5)
tau_var, p_var = kendall_tau(mean_var[pre_tip])
tau_ac1, p_ac1 = kendall_tau(mean_ac1[pre_tip])

print(f"  Variance  tau = {tau_var:+.3f}  p = {p_var:.4f}"
      f"  {'significant' if p_var < 0.05 else 'not significant'}")
print(f"  AC1       tau = {tau_ac1:+.3f}  p = {p_ac1:.4f}"
      f"  {'significant' if p_ac1 < 0.05 else 'not significant'}")

# Bifurcation curves for all elevation bands
gamma_arr  = np.linspace(0.003, 0.011, 400)
bif_curves = [critical_r(gamma_arr, z, acc, K_MELT, T_HORIZON)
              for z, acc in zip(ELEVATIONS, ACCUMULATION)]


# =============================================================================
# 8. FIGURE
# =============================================================================

print("\nBuilding figure...")

DARK   = "#0d1117"
PANEL  = "#161b22"
GRID   = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
BLUE   = "#58a6ff"
RED    = "#f78166"
GREEN  = "#3fb950"
PURPLE = "#d2a8ff"
ORANGE = "#ffa657"
ELEV_C = plt.cm.plasma(np.linspace(0.1, 0.9, len(ELEVATIONS)))

def style(ax, title):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=9)
    for lbl in [ax.xaxis.label, ax.yaxis.label]:
        lbl.set_color(TEXT)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)
    ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT, pad=9)

fig = plt.figure(figsize=(16, 18), facecolor=DARK)
gs  = gridspec.GridSpec(3, 2, figure=fig,
                        hspace=0.44, wspace=0.36,
                        left=0.08, right=0.96,
                        top=0.93,  bottom=0.05)

med_tip = np.nanmedian(onset) if n_tip > 0 else None

# ---- Plot 1: sample trajectories ----------------------------------------
ax1 = fig.add_subplot(gs[0, 0])
style(ax1, f"Stochastic Mass Balance Trajectories  ({FOCUS_ELEV} m)")

for i in range(min(40, N_RUNS)):
    col = RED if not np.isnan(onset[i]) else GREEN
    ax1.plot(YEARS, bal[i], color=col, alpha=0.12, linewidth=0.7)

ax1.fill_between(YEARS, p10_bal, p90_bal,
                 color=BLUE, alpha=0.18, label="10th-90th percentile")
ax1.plot(YEARS, mean_bal, color=BLUE, linewidth=2.2,
         label="Ensemble mean", zorder=5)
ax1.axhline(0, color=RED, linewidth=1.3, linestyle="--",
            alpha=0.85, label="Tipping threshold (B=0)")
ax1.axvline(t_star, color=ORANGE, linewidth=1.2, linestyle="-.", alpha=0.9)
ax1.text(t_star + 1, 0.97,
         f"t* = {t_star:.0f} yr\n(deterministic)",
         transform=ax1.get_xaxis_transform(),
         color=ORANGE, fontsize=8, va="top")

from matplotlib.lines import Line2D
ax1.legend(
    handles=[
        Line2D([0],[0], color=RED,   alpha=0.6, lw=1.5, label="Tipped"),
        Line2D([0],[0], color=GREEN, alpha=0.6, lw=1.5, label="Survived"),
    ] + ax1.get_legend_handles_labels()[0],
    fontsize=8, facecolor=PANEL, labelcolor=TEXT,
    edgecolor=GRID, framealpha=0.85, loc="lower left"
)
ax1.set_xlabel("Year")
ax1.set_ylabel("Annual mass balance  [m w.e.]")

# ---- Plot 2: tipping time distribution ----------------------------------
ax2 = fig.add_subplot(gs[0, 1])
style(ax2, f"Tipping Time Distribution  ({FOCUS_ELEV} m,  N={N_RUNS})")

if n_tip > 5:
    counts, edges = np.histogram(tipped, bins=28)
    centers = 0.5 * (edges[:-1] + edges[1:])
    norm    = (centers - centers.min()) / (centers.max() - centers.min() + 1e-9)
    for left, right, count, f in zip(edges[:-1], edges[1:], counts, norm):
        ax2.bar(left, count, width=(right-left)*0.88,
                color=plt.cm.RdYlGn(f), alpha=0.85, edgecolor=PANEL)

    kde = gaussian_kde(tipped, bw_method=0.35)
    xk  = np.linspace(tipped.min()-3, tipped.max()+3, 400)
    yk  = kde(xk) * n_tip * (edges[1]-edges[0])
    ax2.plot(xk, yk, color=BLUE, linewidth=2.2, label="KDE")

    for pct, col, lbl in [(10, RED, "P10"), (50, ORANGE, "P50"), (90, GREEN, "P90")]:
        v = np.percentile(tipped, pct)
        ax2.axvline(v, color=col, linewidth=1.2, linestyle="--", alpha=0.9)
        ax2.text(v+0.5, ax2.get_ylim()[1]*0.82,
                 f"{lbl}\n{v:.0f}", color=col, fontsize=8)

ax2.axvline(t_star, color=ORANGE, linewidth=1.4, linestyle="-.", alpha=0.9,
            label=f"t* = {t_star:.0f} yr (analytical)")
ax2.text(0.96, 0.96,
         f"Tipped   : {n_tip} / {N_RUNS}\nSurvived : {n_surv} / {N_RUNS}",
         transform=ax2.transAxes, ha="right", va="top",
         color=TEXT, fontsize=9,
         bbox=dict(boxstyle="round", facecolor=PANEL,
                   edgecolor=GRID, alpha=0.9))
ax2.set_xlabel("Year of tipping")
ax2.set_ylabel("Number of runs")
ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

# ---- Plot 3: bifurcation diagram ----------------------------------------
ax3 = fig.add_subplot(gs[1, 0])
style(ax3, "Bifurcation Diagram  (r, gamma) Parameter Space")

for i, (z, acc, col) in enumerate(zip(ELEVATIONS, ACCUMULATION, ELEV_C)):
    rc = bif_curves[i]
    ok = np.isfinite(rc) & (rc > 0) & (rc < 0.20)
    if ok.any():
        ax3.plot(gamma_arr[ok], rc[ok], color=col, linewidth=2.0,
                 label=f"{z} m")

ax3.scatter([GAMMA], [R_SCENARIO], color="white", s=150, zorder=10,
            edgecolors=RED, linewidths=2.5, label="Scenario climate")
ax3.text(GAMMA + 0.0001, R_SCENARIO + 0.004,
         f"Scenario\nr={R_SCENARIO}", color=TEXT, fontsize=8)
ax3.text(0.007,  0.025, "STABLE",   color=GREEN, fontsize=10,
         fontweight="bold", alpha=0.9)
ax3.text(0.0045, 0.155, "COLLAPSE", color=RED,   fontsize=10,
         fontweight="bold", alpha=0.9)
ax3.set_xlabel("Lapse rate  gamma  [deg C / m]")
ax3.set_ylabel("Warming rate  r  [deg C / year]")
ax3.set_ylim(0, 0.19)
ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT,
           edgecolor=GRID, ncol=2, loc="upper left")

# ---- Plot 4: phase portrait ----------------------------------------------
ax4 = fig.add_subplot(gs[1, 1])
style(ax4, f"Phase Portrait  (Year 0 vs Year {T_HORIZON})  ({FOCUS_ELEV} m)")

b0 = bal[:, 0]
bf = bal[:, -1]
tip_norm = np.where(
    np.isnan(onset), 1.0,
    (onset - np.nanmin(onset)) / (np.nanmax(onset) - np.nanmin(onset) + 1e-9)
)
sc = ax4.scatter(b0, bf, c=tip_norm, cmap="RdYlGn",
                 s=14, alpha=0.55, linewidths=0)
cb = plt.colorbar(sc, ax=ax4)
cb.set_label("Tipping time  (green = survived / late)", color=TEXT, fontsize=9)
cb.ax.yaxis.set_tick_params(color=TEXT)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)

ax4.axhline(0, color=RED, linewidth=1.0, linestyle="--", alpha=0.6)
ax4.axvline(0, color=RED, linewidth=1.0, linestyle="--", alpha=0.6)
ax4.set_xlabel("Mass balance at year 0  [m w.e.]")
ax4.set_ylabel(f"Mass balance at year {T_HORIZON}  [m w.e.]")
ax4.text(0.02, 0.97, "Stable  ->  Stable",
         transform=ax4.transAxes, color=GREEN, fontsize=8, va="top")
ax4.text(0.02, 0.06, "Stable  ->  Collapsed",
         transform=ax4.transAxes, color=RED, fontsize=8, va="bottom")

# ---- Plot 5: rising variance --------------------------------------------
ax5 = fig.add_subplot(gs[2, 0])
style(ax5, f"Early Warning: Rising Variance  ({FOCUS_ELEV} m)")

vp25 = np.nanpercentile(var_runs, 25, axis=0)
vp75 = np.nanpercentile(var_runs, 75, axis=0)

ax5.fill_between(YEARS, vp25, vp75,
                 color=ORANGE, alpha=0.20, label="IQR across runs")
ax5.plot(YEARS, mean_var, color=ORANGE, linewidth=2.2,
         label="Ensemble mean variance")
ax5.axvspan(EWS_WINDOW, t_star - 5, alpha=0.08, color=BLUE,
            label="Pre-tipping analysis window")
ax5.axvline(t_star, color=RED, linewidth=1.2,
            linestyle="-.", alpha=0.85, label=f"t* = {t_star:.0f} yr")

sig = "significant" if p_var < 0.05 else "not significant"
ax5.text(0.03, 0.95,
         f"Kendall tau = {tau_var:+.3f}\np = {p_var:.4f}  ({sig})",
         transform=ax5.transAxes, va="top", color=ORANGE, fontsize=9,
         bbox=dict(boxstyle="round", facecolor=PANEL,
                   edgecolor=GRID, alpha=0.9))
ax5.set_xlabel("Year")
ax5.set_ylabel(f"Rolling variance  (window={EWS_WINDOW} yr)")
ax5.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

# ---- Plot 6: rising AC1 -------------------------------------------------
ax6 = fig.add_subplot(gs[2, 1])
style(ax6, f"Early Warning: Rising AC1  ({FOCUS_ELEV} m)")

ap25 = np.nanpercentile(ac1_runs, 25, axis=0)
ap75 = np.nanpercentile(ac1_runs, 75, axis=0)

ax6.fill_between(YEARS, ap25, ap75,
                 color=PURPLE, alpha=0.20, label="IQR across runs")
ax6.plot(YEARS, mean_ac1, color=PURPLE, linewidth=2.2,
         label="Ensemble mean AC1")
ax6.axhline(1.0, color=RED, linewidth=0.8, linestyle=":",
            alpha=0.6, label="AC1 = 1  (bifurcation limit)")
ax6.axvspan(EWS_WINDOW, t_star - 5, alpha=0.08, color=BLUE,
            label="Pre-tipping analysis window")
ax6.axvline(t_star, color=RED, linewidth=1.2,
            linestyle="-.", alpha=0.85, label=f"t* = {t_star:.0f} yr")

sig = "significant" if p_ac1 < 0.05 else "not significant"
ax6.text(0.03, 0.97,
         f"Kendall tau = {tau_ac1:+.3f}\np = {p_ac1:.4f}  ({sig})",
         transform=ax6.transAxes, va="top", color=PURPLE, fontsize=9,
         bbox=dict(boxstyle="round", facecolor=PANEL,
                   edgecolor=GRID, alpha=0.9))
ax6.set_xlabel("Year")
ax6.set_ylabel(f"Rolling lag-1 autocorrelation  (window={EWS_WINDOW} yr)")
ax6.set_ylim(-0.6, 1.15)
ax6.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, edgecolor=GRID)

# ---- title and subtitle --------------------------------------------------
fig.text(0.5, 0.975,
         "Himalayan Glacier Mass Balance: Stochastic Bifurcation and Early Warning Signals",
         ha="center", va="top", fontsize=14, fontweight="bold", color=TEXT)
fig.text(0.5, 0.958,
         f"Focus: {FOCUS_ELEV} m  |  OU noise: theta={THETA}, sigma={SIGMA} C  |"
         f"  r={R_SCENARIO} C/yr  |  N={N_RUNS} Monte Carlo runs  |"
         f"  Analytical t* = {t_star:.0f} yr",
         ha="center", va="top", fontsize=9.5, color=MUTED)

out_img = "/mnt/user-data/outputs/glacier_stochastic.png"
plt.savefig(out_img, dpi=150, bbox_inches="tight", facecolor=DARK)
plt.close()
print(f"Figure saved -> {out_img}")

import shutil
shutil.copy(__file__, "/mnt/user-data/outputs/glacier_stochastic.py")
print(f"Script saved -> /mnt/user-data/outputs/glacier_stochastic.py")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Focus band     : {FOCUS_ELEV} m")
print(f"  Analytical t*  : {t_star:.1f} yr")
print(f"  Runs tipped    : {n_tip} / {N_RUNS}")
if n_tip > 0:
    print(f"  Median tip     : {np.median(tipped):.0f} yr")
    print(f"  80% interval   : [{np.percentile(tipped,10):.0f}, "
          f"{np.percentile(tipped,90):.0f}] yr")
print(f"  Variance tau   : {tau_var:+.3f}  (p={p_var:.4f})")
print(f"  AC1 tau        : {tau_ac1:+.3f}  (p={p_ac1:.4f})")
