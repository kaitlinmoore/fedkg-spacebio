"""Deterministic synthetic binary-classification data split across 3 simulated sites.

AI-drafted (Claude, WS-02). This is a test bed for checking FedAvg against FLARE. It is **not**
real data and not a model of OSDR. Labels are just 0/1 here; in the project they would be
flight vs. ground.

Design: why the sites differ
----------------------------
A FedAvg check where every site looks the same would pass even with the wrong client weights,
because the average of near-identical models is the same under any weighting. So the sites are
built to differ along three axes:

1. **Unequal sizes** (400 / 150 / 50). Sample-count weighting and equal weighting then give
   visibly different global models.
2. **Feature shift (covariate shift).** Each site's features are centered on a different mean, a
   crude stand-in for between-study batch effects.
3. **Label shift.** Each site gets its own intercept offset, so the positive rate differs
   (52.5% / 82.0% / 12.0% with seed 0). Every site still has both classes; the generator checks
   this, mirroring the project rule that a site needs both arms.

All sites share the same true coefficients ``TRUE_COEF``: the "biology" is common and the
site-specific parts are nuisance. Labels are drawn from Bernoulli(sigmoid(·)), so the data is
noisy and not linearly separable, and unpenalized logistic regression has a finite optimum.

Determinism and independence
----------------------------
Site i's data comes from ``np.random.default_rng([seed, i])``: one independent stream per site.
A FLARE client can therefore generate *only its own site's* data from (seed, site name), with no
access to other sites. That keeps the federation boundary honest even with synthetic data. It
also means changing one site's size doesn't change the other sites' samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_FEATURES = 4
# Shared "true" effect sizes. The last feature is pure noise (coefficient 0).
TRUE_COEF = np.array([1.5, -1.0, 0.5, 0.0])
TRUE_INTERCEPT = 0.0


@dataclass(frozen=True)
class SiteSpec:
    n_samples: int
    feature_shift: tuple[float, ...]  # added to the N(0, I) features: covariate shift
    intercept_shift: float  # added to the logit: changes the site's positive rate


# Site names match FLARE's SimEnv defaults (site-1, site-2, ...), so a client can look itself up.
SITE_SPECS: dict[str, SiteSpec] = {
    "site-1": SiteSpec(n_samples=400, feature_shift=(0.0, 0.0, 0.0, 0.0), intercept_shift=0.0),
    "site-2": SiteSpec(n_samples=150, feature_shift=(1.0, 0.0, -0.5, 0.0), intercept_shift=1.0),
    "site-3": SiteSpec(n_samples=50, feature_shift=(-1.0, 1.0, 0.0, 0.5), intercept_shift=-1.0),
}
SITE_NAMES: tuple[str, ...] = tuple(SITE_SPECS)


@dataclass(frozen=True)
class SiteData:
    name: str
    X: np.ndarray  # (n_samples, N_FEATURES), float64
    y: np.ndarray  # (n_samples,), float64 in {0., 1.}

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])


def make_site_data(site_name: str, seed: int = 0) -> SiteData:
    """Generate one site's data from (seed, site name), touching no other site."""
    if site_name not in SITE_SPECS:
        raise KeyError(f"unknown site {site_name!r}; expected one of {SITE_NAMES}")
    spec = SITE_SPECS[site_name]
    site_index = SITE_NAMES.index(site_name)
    rng = np.random.default_rng([seed, site_index])  # independent stream per site

    X = rng.standard_normal((spec.n_samples, N_FEATURES)) + np.asarray(spec.feature_shift)
    logits = X @ TRUE_COEF + TRUE_INTERCEPT + spec.intercept_shift
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(spec.n_samples) < p).astype(np.float64)  # Bernoulli(p)

    if y.min() == y.max():
        # One-class site: the model would learn the site, not the signal. Fail loudly.
        raise ValueError(f"{site_name} (seed={seed}) has only class {y[0]:.0f}")
    return SiteData(name=site_name, X=X, y=y)


def make_all_sites(seed: int = 0) -> list[SiteData]:
    """All sites, in SITE_NAMES order. Only reference and centralized code may call this."""
    return [make_site_data(name, seed) for name in SITE_NAMES]


def pool(sites: list[SiteData]) -> tuple[np.ndarray, np.ndarray]:
    """Stack all sites' data into one (X, y): the centralized baseline that federation can't use."""
    return np.vstack([s.X for s in sites]), np.concatenate([s.y for s in sites])
