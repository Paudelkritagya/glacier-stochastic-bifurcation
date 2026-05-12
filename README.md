# glacier-stochastic-bifurcation

Himalayan glacier mass balance model with stochastic temperature forcing,
Monte Carlo simulation, and early warning signal detection.

## What this is

A glacier survives as long as annual snowfall exceeds annual melt. As the
climate warms, that balance tips. This project models that process for a
Nepal elevation transect (Chitwan at 300 m up to Mustang at 5500 m) and
asks a question the standard deterministic model cannot answer: can you
detect a glacier's approaching collapse before it actually happens, just
from the statistics of its year-to-year fluctuations?

The answer is yes. Near a tipping point, two things happen to any noisy
dynamical system. The variance of its fluctuations rises, because the
restoring force is weakening. The lag-1 autocorrelation also rises, because
recovery from bad years gets slower. Both signals appear well before the
actual collapse. This phenomenon is called critical slowing down, and it
shows up clearly in the model output: variance Kendall $\tau$ = +0.98
($p$ < 0.0001), autocorrelation Kendall $\tau$ = +0.75 ($p$ < 0.0001).

The same mathematics governs early warning signals in financial markets.
A market tipping from a calm regime into a volatile one is a stochastic
bifurcation described by the same equations as a glacier tipping into
irreversible melt. The noise model used here, the Ornstein-Uhlenbeck
process, is the same process used in the Vasicek interest rate model in
quantitative finance.

## Files

**glacier_stochastic.py** is the main file. It models temperature as a
mean-reverting stochastic process, runs 500 Monte Carlo simulations, derives
the bifurcation boundary analytically, and computes rolling variance and
autocorrelation signals across the ensemble. Six plots are produced covering
sample trajectories, tipping time distribution, bifurcation diagram, phase
portrait, and both early warning signals with Kendall $\tau$ statistics.

**glacier_sensitivity.py** is the deterministic foundation. It sweeps each
of the three model parameters across a physically plausible range while
holding the others fixed, and records melt-onset year and cumulative mass
loss for each elevation band. The main result is that the lapse rate
$\gamma$ is the critical parameter protecting high-altitude glaciers:
reducing it warms the air at elevation faster than increasing the warming
rate alone, which collapses the safety margin from below.

## How to run

```bash
pip install numpy matplotlib scipy
python glacier_sensitivity.py
python glacier_stochastic.py
```

Both scripts save figures and a copy of themselves to the working directory.
Runtime is under a minute on a standard laptop.

## The model

Annual mass balance at elevation $z$ in year $t$:

$$B(z,\, t) = \text{Acc}(z) - k \cdot 365 \cdot \max\!\bigl(T_{\text{eff}}(z,t),\ 0\bigr)$$

Effective temperature with stochastic forcing:

$$T_{\text{eff}}(z,\, t) = T_{\text{base}} + r\,t + \eta(t) - \gamma\, z$$

where $\eta(t)$ follows an Ornstein-Uhlenbeck process:

$$d\eta = -\theta\,\eta\;dt + \sigma\;dW$$

The exact discrete update used in the code (Gillespie 1996):

$$\eta(t+1) = e^{-\theta}\,\eta(t) + \sigma\sqrt{\frac{1 - e^{-2\theta}}{2\theta}}\;\varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0,1)$$

The analytical tipping year (deterministic skeleton, $\eta = 0$), derived
by setting $B = 0$ and solving for $t$:

$$t^*(z) = \frac{\dfrac{\text{Acc}(z)}{k \cdot 365} + \gamma\, z - T_{\text{base}}}{r}$$

Setting $t^* = T_{\text{horizon}}$ and solving for $r$ gives the
bifurcation boundary in $(r,\, \gamma)$ space:

$$r^*(z,\, \gamma) = \frac{\dfrac{\text{Acc}(z)}{k \cdot 365} + \gamma\, z - T_{\text{base}}}{T_{\text{horizon}}}$$

For the 4800 m focus band at $r = 0.12$ °C/yr:

$$t^* = \frac{6.702}{0.12} = 55.9 \text{ yr}$$

The Monte Carlo median is 58 years, confirming the simulation implements
the math correctly.

## Early warning signal theory

Near a bifurcation, the dominant eigenvalue $\lambda$ of the linearised
system approaches zero. By the fluctuation-dissipation theorem:

$$\text{Var}(B) \;\sim\; \frac{\sigma^2}{2\,|\lambda|} \;\longrightarrow\; \infty \quad \text{as } \lambda \to 0$$

$$\text{AC1}(B) \;=\; e^{\lambda} \;\longrightarrow\; 1 \quad \text{as } \lambda \to 0$$

Both diverge at the tipping point. Kendall's $\tau$ tests whether each
signal has a statistically significant rising trend in the pre-tipping
window:

$$\tau = \frac{(\text{concordant pairs}) - (\text{discordant pairs})}{\text{total pairs}}$$

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $k$ | 0.006 m w.e. / °C / day | degree-day melt factor |
| $r$ | 0.12 °C / yr | warming rate (scenario) |
| $\gamma$ | 0.0065 °C / m | atmospheric lapse rate |
| $\theta$ | 0.4 / yr | OU mean-reversion speed |
| $\sigma$ | 1.2 °C | OU noise amplitude |
| $N$ | 500 | Monte Carlo ensemble size |

## References

Scheffer et al. (2009). Early-warning signals for critical transitions.
*Nature*, 461, 53-59.

Dakos et al. (2008). Slowing down as an early warning signal for abrupt
climate change. *PNAS*, 105(38), 14308-14312.

Vasicek, O. (1977). An equilibrium characterization of the term structure.
*Journal of Financial Economics*, 5(2), 177-188.
