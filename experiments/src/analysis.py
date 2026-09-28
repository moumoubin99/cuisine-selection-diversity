"""The audit statistics (A1-A5), each stated so that it is not true by construction.

Selecting the top k of N reduces diversity for *any* scorer, including a random
one.  Every quantity here is therefore defined against a matched null, or as a
shape/direction that a content-blind selector would not produce.
"""

from __future__ import annotations

import math

import numpy as np

from .cube_metrics import cultural_diversity_strict, coverage, simpson_diversity

__all__ = [
    "excess_loss", "dose_response", "stereotype_shift", "per_country_price",
    "MAX_NULL_INSUFFICIENT", "MAX_NULL_BIAS_LOG", "null_bias_bounds",
    "equivalence_test_country_marginal",
    "convergence", "bootstrap_ci", "pearson_with_ci",
]


def _div(labels):
    return cultural_diversity_strict(labels, batch_size=min(8, max(1, len(labels))))


def _div_at(labels, m, n_sub=64, rng=None):
    """Diversity of `labels` evaluated at a FIXED set size `m`.

    The CUBE metric normalises the Vendi score by the batch size, so
    `VS(S)/|S|` is **not comparable across different |S|**: a fully collapsed
    set of 8 identical labels scores 1/8 = 0.125 while a fully collapsed set of
    4 scores 1/4 = 0.25.  Comparing top-8 against top-4 directly would show
    "diversity rising with selection pressure" as a pure artefact of the
    normaliser.

    Every cross-k comparison therefore evaluates at one common size `m`
    (<= the smallest k in the sweep) by averaging the metric over `n_sub`
    random subsamples of that size.  Pinned by
    `test_fixed_size_evaluation_removes_the_normaliser_artefact`.
    """
    rng = rng or np.random.default_rng(0)
    labels = list(labels)
    m = int(min(m, len(labels)))
    if m < 2:
        return float("nan")
    if m == len(labels):
        return cultural_diversity_strict(labels, batch_size=m)
    idx = np.arange(len(labels))
    vals = [cultural_diversity_strict([labels[i] for i in rng.choice(idx, size=m, replace=False)],
                                      batch_size=m)
            for _ in range(n_sub)]
    return float(np.mean(vals))


# ---------------------------------------------------------------- A1: excess loss

def excess_loss(pool_labels, selected_labels, null_selected_labels):
    """Diversity lost by the scorer *beyond* what a matched null selector loses.

    Args:
      pool_labels: labels of all N samples.
      selected_labels: labels of the k the scorer kept.
      null_selected_labels: list of label-lists, one per draw of the matched
        content-blind null selector (same N, same k).

    Returns a dict with the pool, scorer and null diversities, the raw loss,
    and the excess loss.  `excess` > 0 means the scorer removed cultural
    concepts that selection pressure alone does not explain.
    """
    d_pool = _div(pool_labels)
    d_sel = _div(selected_labels)
    nulls = np.array([_div(l) for l in null_selected_labels], dtype=np.float64)
    d_null = float(nulls.mean()) if nulls.size else float("nan")
    return {
        "diversity_pool": d_pool,
        "diversity_selected": d_sel,
        "diversity_null_mean": d_null,
        "diversity_null_std": float(nulls.std(ddof=1)) if nulls.size > 1 else 0.0,
        "raw_loss": d_pool - d_sel,
        "excess_loss": d_null - d_sel,
        "null_z": ((d_null - d_sel) / nulls.std(ddof=1)) if nulls.size > 1 and nulls.std(ddof=1) > 0 else float("nan"),
    }


#: If more than this fraction of the matched null draws fall below the common
#: count, the surviving draws are a biased sample of the null and the contrast
#: is reported non-estimable rather than computed on the survivors.
#:
#: Round-2 code review, P1: this was 0.20, and 0.20 is not a small number here.
#: The reviewer's exact enumeration -- pool `aaabbbbbb`, k=4, m=3, the last two
#: images unresolved, top-k = [0,1,2,7] -- excludes 21 of the 126 null subsets
#: (16.67%, comfortably under the old threshold) and reports -42.4% where the
#: complete-label answer is +1.6%.  The excluded draws are exactly the
#: kept-poor ones, so survivorship, not selection, produced the finding.
#:
#: The threshold is now set from a bound rather than from taste.  An excluded
#: draw's rarefied diversity is unknown but lies in [1, m], so the
#: unconditional null mean log-diversity is bracketed within `frac * log(m)`
#: and the contrast inherits that bracket (`delta_bounds`, computed by
#: `null_bias_bounds`).  At frac <= 0.02 and m = 8 the bracket is 0.042 log
#: units, under half the pre-registered margin log(1.10) = 0.0953; a pool whose
#: bracket exceeds the margin is non-estimable whatever the fraction, because a
#: contrast that cannot be placed on one side of the margin is not a result.
MAX_NULL_INSUFFICIENT = 0.02

#: The bracket a single pool's contrast may carry from unusable null draws,
#: in log units.  Equal to the pre-registered 10% margin.
MAX_NULL_BIAS_LOG = float(np.log1p(0.10))


def null_bias_bounds(log_d_selected, log_null_survivors, frac, m):
    """Assumption-free bracket for the contrast the DISCARDED draws would give.

    The diversity is normalised (`exp(H) / m`, see `_exact_match_rarefied`), so a
    log value lies in `[-log m, 0]`.  The null mean is `(1-f)*L_surv + f*L_excl`
    with `L_excl` in that range, and the contrast lies in an interval of width
    `f * log(m)`: its LOWER end puts every excluded draw at the maximum
    (`L_excl = 0`), its upper end at the minimum (`L_excl = -log m`).  Neither
    end is the survivors-only contrast unless `f = 0`.  With `f = 0`
    the interval collapses onto the point estimate, which is the only case in
    which the point estimate needs no caveat.

    Experiment audit, 2026-09-24: the bracket used to assume `L_excl` in
    `[0, log m]` (unnormalised log Vendi), which shifted it down by
    `f * log m`; its width, and hence every non-estimability decision, was
    already right.
    """
    f = float(frac)
    l_surv = float(log_null_survivors)
    lo = float(log_d_selected) - (1.0 - f) * l_surv
    return (lo, lo + f * float(np.log(max(int(m), 1))))


# ------------------------------------------------------------- A2: dose response

def dose_response(pool_labels, scores, ks, null_fn, n_null=50, rng=None,
                  eval_size=None, keep=None):
    """Excess loss as a function of selection pressure log(N/k).

    A monotone increase in excess loss with pressure is the signature of a
    *taste* direction.  A flat or non-monotone (inverted-U) response is the
    signature of fidelity preference (arXiv 2608.23593) and would refute the
    taste interpretation -- which is why this curve, not a single k, is the
    primary figure.

    `null_fn(k, rng) -> indices` supplies the matched null.

    `keep` is the label filter, and it is applied **after** selection, over the
    whole pool of N.  Round-2 code review, P2: the caller used to hand this
    function an already-filtered pool, which restores at every point of the
    dose curve precisely the filter-first intervention that `_endpoint` was
    fixed to avoid -- "the best of the images we judged usable" rather than
    "the images the deployed system would ship".  The reviewer's reproduction
    reported N=4, a selected diversity of 1 and zero excess loss on a pool
    whose raw endpoint is correctly non-estimable with no kept selected image
    at all.  Every k now carries its own `n_kept_selected`, its own
    `null_insufficient_frac`, and its own non-estimability.
    """
    rng = rng or np.random.default_rng(0)
    labels = np.asarray(pool_labels, dtype=object)
    scores = np.asarray(scores, dtype=np.float64)
    n = len(labels)
    keep = [True] * n if keep is None else [bool(x) for x in keep]
    if len(keep) != n:
        raise ValueError("keep must have one entry per pool image")
    order = np.argsort(-scores, kind="stable")
    ks = [int(k) for k in ks if 2 <= int(k) <= n]
    if not ks:
        return []
    # One common evaluation size for every k, so the batch-size normaliser
    # cannot manufacture a trend.  See `_div_at`.
    m = eval_size or min(ks)
    pool_kept = [labels[i] for i in range(n) if keep[i]]
    d_pool = (_div_at(pool_kept, m, rng=rng) if len(pool_kept) >= m
              else float("nan"))
    rows = []
    for k in ks:
        sel = [labels[i] for i in order[:k] if keep[i]]
        d_null, insufficient = [], 0
        for _ in range(n_null):
            idx = np.asarray(null_fn(k, rng), dtype=int)
            nl = [labels[i] for i in idx if keep[i]]
            if len(nl) < m:
                insufficient += 1
            else:
                d_null.append(_div_at(nl, m, rng=rng))
        frac = insufficient / float(n_null) if n_null else 1.0
        row = {
            "k": k, "N": n, "eval_size": m,
            "pressure": float(np.log(n / k)),
            "n_kept_pool": int(sum(keep)),
            "n_kept_selected": len(sel),
            "selected_deficit": 1.0 - len(sel) / float(k),
            "null_insufficient_frac": frac,
            "diversity_pool": d_pool,
        }
        if len(sel) < m:
            rows.append({**row, "non_estimable": True,
                         "reason": f"selected set has {len(sel)} < m={m} kept images"})
            continue
        if frac > MAX_NULL_INSUFFICIENT or not d_null:
            rows.append({**row, "non_estimable": True,
                         "reason": f"{frac:.0%} of null draws fell below the "
                                   f"common count (> {MAX_NULL_INSUFFICIENT:.0%}); "
                                   "the survivors are a biased reference"})
            continue
        d_sel = _div_at(sel, m, rng=rng)
        d_null = np.asarray(d_null, dtype=np.float64)
        rows.append({
            **row,
            "non_estimable": False,
            "diversity_selected": d_sel,
            "diversity_null_mean": float(d_null.mean()),
            "diversity_null_std": float(d_null.std(ddof=1)) if d_null.size > 1 else 0.0,
            "excess_loss": float(d_null.mean() - d_sel),
            # The primary contrast's form (`_endpoint`): log selected minus the
            # mean LOG null.  Added after experiment audit run 02, which found
            # the dose summary used log(selected / arithmetic null mean) and so
            # could not be compared with the k = 16 contrast.
            "log_contrast": (float(np.log(d_sel) - np.log(d_null).mean())
                             if d_sel > 0 and np.all(d_null > 0) else float("nan")),
            "null_z": float((d_null.mean() - d_sel) / d_null.std(ddof=1))
                      if d_null.size > 1 and d_null.std(ddof=1) > 0 else float("nan"),
        })
    return rows


# ------------------------------------------------------------ A3: stereotype shift

def stereotype_shift(pool_labels, selected_labels, prototypicality):
    """Do survivors concentrate on the more prototypical artifacts?

    `prototypicality` maps artifact label -> score (the text-prototypicality
    proxy from `data.TEXT_PROTOTYPICALITY`).  Reported as the shift in mean
    prototypicality from pool to selected, standardised by the pool's own
    spread, so it is comparable across countries with different vocabularies.
    A content-blind null has expected shift 0.
    """
    def m(labels):
        v = [prototypicality[l] for l in labels if l in prototypicality]
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))

    mu_p, sd_p, n_p = m(pool_labels)
    mu_s, _, n_s = m(selected_labels)
    return {
        "pool_mean_prototypicality": mu_p,
        "selected_mean_prototypicality": mu_s,
        "shift": mu_s - mu_p,
        "standardised_shift": (mu_s - mu_p) / sd_p if sd_p > 0 else float("nan"),
        "n_pool_matched": n_p,
        "n_selected_matched": n_s,
    }


# ---------------------------------------------------------- A4: per-country price

def per_country_price(per_country_rows):
    """Rank countries by excess loss at fixed k/N -- the 'price of aesthetics'.

    `per_country_rows` maps country -> the dict returned by `excess_loss`.
    Heterogeneity is what makes the audit more than a re-run of an existing
    demographic audit, so the spread is reported explicitly along with a
    permutation test that the spread exceeds what null variation produces.
    """
    countries = sorted(per_country_rows)
    vals = np.array([per_country_rows[c]["excess_loss"] for c in countries])
    return {
        "countries": countries,
        "excess_loss": vals.tolist(),
        "spread": float(vals.max() - vals.min()) if vals.size else float("nan"),
        "ranking": [countries[i] for i in np.argsort(-vals)],
        "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
    }


# ------------------------------------------------------------- A5: convergence

def convergence(country_label_sets_before, country_label_sets_after):
    """Does selection move different countries' outputs toward each other?

    Measured as mean pairwise Jaccard overlap of the artifact label sets across
    countries, before vs after selection.  An increase means the countries'
    depictions became more alike -- cross-cultural homogenisation, which is a
    strictly stronger claim than each country losing diversity on its own.
    """
    def mean_pairwise_jaccard(sets):
        cs = sorted(sets)
        vals = []
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = set(sets[cs[i]]), set(sets[cs[j]])
                if not (a or b):
                    continue
                vals.append(len(a & b) / len(a | b))
        return float(np.mean(vals)) if vals else float("nan")

    before = mean_pairwise_jaccard(country_label_sets_before)
    after = mean_pairwise_jaccard(country_label_sets_after)
    return {"jaccard_before": before, "jaccard_after": after, "delta": after - before}


# --------------------------------------------------------------------- inference

def bootstrap_ci(values, statistic=np.mean, n_boot=10000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for a statistic of a 1-D sample."""
    rng = rng or np.random.default_rng(0)
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = statistic(v[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(v)), float(lo), float(hi)


def pearson_with_ci(x, y, alpha=0.05):
    """Pearson r with a Fisher-z confidence interval.

    This is the arbiter for the mechanism claim: r > 0.7 between predicted and
    observed per-country loss.  With only 8 countries the CI is wide, which is
    the honest reason the pre-registered threshold is on the point estimate AND
    the CI lower bound must exclude 0.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.size
    if n < 4:
        return {"r": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(n)}
    r = float(np.corrcoef(x, y)[0, 1])
    r_c = min(max(r, -0.999999), 0.999999)
    z = np.arctanh(r_c)
    se = 1.0 / np.sqrt(n - 3)
    from math import erf, sqrt
    # two-sided normal quantile without scipy
    def _ppf(p):
        # Acklam's rational approximation, plenty accurate for CI endpoints
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        pl, ph = 0.02425, 1 - 0.02425
        if p < pl:
            q = sqrt(-2 * np.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > ph:
            q = sqrt(-2 * np.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r2 = q * q
        return (((((a[0]*r2+a[1])*r2+a[2])*r2+a[3])*r2+a[4])*r2+a[5])*q / (((((b[0]*r2+b[1])*r2+b[2])*r2+b[3])*r2+b[4])*r2+1)

    zc = _ppf(1 - alpha / 2)
    lo, hi = np.tanh(z - zc * se), np.tanh(z + zc * se)
    return {"r": r, "lo": float(lo), "hi": float(hi), "n": int(n)}


# ==========================================================================
# Revisions demanded by the Phase-4 critical review (idea-stage/reviews/
# critical_review.md).  These replace, not supplement, the earlier estimators
# where they overlap; the originals are kept only for the unit tests that
# document why they were wrong.
# ==========================================================================

def log_contrast(d_selected, d_null_draws):
    """R2: the pool-level contrast, on the log scale.

        delta = log D_top  -  mean_r log D_random_r

    The reviewer's point is that the raw difference `D_null - D_sel` is not the
    quantity anyone wants to pool across countries whose diversities differ by
    7x: an absolute loss of 0.1 means something very different against a pool
    of 200 artifacts than against one of 24.  `exp(delta) - 1` is the
    proportional change against the random geometric mean, which is comparable
    and is what the 10% pre-registered threshold refers to.
    """
    d_selected = float(d_selected)
    draws = np.asarray(d_null_draws, dtype=np.float64)
    if d_selected <= 0 or draws.size == 0 or np.any(draws <= 0):
        return {"delta": float("nan"), "proportional_change": float("nan"),
                "n_null": int(draws.size), "non_estimable": True}
    delta = float(np.log(d_selected) - np.log(draws).mean())
    return {
        "delta": delta,
        "proportional_change": float(np.expm1(delta)),
        "log_null_mean": float(np.log(draws).mean()),
        "log_null_sd": float(np.log(draws).std(ddof=1)) if draws.size > 1 else 0.0,
        "n_null": int(draws.size),
        "non_estimable": False,
    }


def pool_contrasts_by_country(pool_rows):
    """Group pool-level deltas by country.  The unit of replication is a POOL.

    `pool_rows` is an iterable of dicts with keys 'country' and 'delta'.
    Images are not independent observations and neither are rarefaction draws;
    only a freshly generated candidate pool is.  Everything downstream of here
    counts pools.
    """
    by = {}
    for r in pool_rows:
        if r.get("non_estimable") or not np.isfinite(r.get("delta", np.nan)):
            continue
        by.setdefault(r["country"], []).append(float(r["delta"]))
    return {c: np.asarray(v, dtype=np.float64) for c, v in by.items()}


def _student_t_ppf(p, df):
    """Exact Student-t quantile by bisection on the CDF.

    No scipy in this repository, and the Cornish-Fisher expansion is not
    accurate enough at the df=3..7 this study actually runs at, which is
    precisely where a mis-calibrated quantile turns an inconclusive result into
    a claimed one.
    """
    df = float(df)
    p = float(p)
    if df <= 0 or not (0.0 < p < 1.0):
        if p <= 0.0:
            return float("-inf")
        if p >= 1.0:
            return float("inf")
        return float("nan")
    # Expand the bracket until it actually contains the quantile.  A FIXED
    # [-1e3, 1e3] bracket does not merely lose precision in the far tail, it
    # silently returns the bracket endpoint: at df=3, t(1 - 1e-10) is 2225.77
    # and a fixed bracket reports 1000.  The study uses alpha=0.05 where the
    # difference is nil, but a clipped quantile is a wrong answer reported as a
    # right one, which is the failure mode this file exists to avoid.
    hi = 1.0
    while _student_t_cdf(hi, df) < p and hi < 1e300:
        hi *= 4.0
    lo = -1.0
    while _student_t_cdf(lo, df) > p and lo > -1e300:
        lo *= 4.0
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if not np.isfinite(mid):
            break
        if _student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _student_t_cdf(t, df):
    x = df / (df + t * t)
    ib = _betai(0.5 * df, 0.5, x)
    return 1.0 - 0.5 * ib if t > 0 else 0.5 * ib


def _betai(a, b, x):
    """Regularised incomplete beta I_x(a, b) -- Lentz continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def _betacf(a, b, x, itmax=300, eps=3e-14, fpmin=1e-300):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _f_sf(x, df1, df2):
    """Upper tail of the F distribution, via the incomplete beta."""
    x = float(x)
    if np.isnan(x):
        return float("nan")
    if x <= 0:
        return 1.0
    if np.isinf(x):
        return 0.0          # NOT 1.0: the upper tail beyond +inf is empty
    return _betai(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * x))


#: Welch's test is mildly LIBERAL at small group sizes.  Measured on this
#: design (8 countries, heteroscedastic normal pools, nominal 5%): 9.2% at 4
#: pools/country, 6.8% at 6, 6.1% at 8, 5.3% at 16.  Below this threshold the
#: result is flagged and must not carry a heterogeneity claim on its own.
#: Pinned by `test_heterogeneity_size_is_measured_not_assumed`.
MIN_POOLS_FOR_HETEROGENEITY = 8


def heterogeneity_test(by_country, n_perm=2000, rng=None):
    """A4 under the EQUAL-EFFECTS null, allowing unequal country variances.

    The design reviewer rejected two shortcuts explicitly: permuting scores
    within pools tests "no score/content association", a weaker null that
    cannot distinguish equal-but-nonzero country effects from unequal ones;
    and permuting country labels assumes exchangeable variances, false at a 7x
    vocabulary spread.  What is needed is a test of equal means that tolerates
    unequal variances -- which is Welch's ANOVA, and that is the primary test
    here:

        F = [sum_c w_c (m_c - m_bar)^2 / (K-1)] / (1 + 2(K-2)L/3),
        w_c = n_c / s_c^2,  L = 3/(K^2-1) * sum_c (1 - w_c/W)^2 / (n_c - 1)

    referred to F(K-1, 1/L).

    Two bootstrap variants were built and MEASURED before this one was chosen,
    because the code review had found the original severely mis-calibrated:

    * re-centre and resample within country, statistic studentised by the
      *observed* variances -- 33% rejection at a nominal 5% (a resample's mean
      moves less than the observed mean does relative to a fixed s_c);
    * the same resampling with the variances re-estimated inside each
      resample -- 0.0% at 4 pools and 1.5% at 8 (an occasional near-constant
      resample produces an enormous null statistic and swallows the tail).

    Welch's own reference distribution sits between them and is only mildly
    liberal (see `MIN_POOLS_FOR_HETEROGENEITY`).  The conservative bootstrap is
    still returned as `p_value_bootstrap_conservative`, and the study's
    decision rule requires BOTH a Welch rejection and at least one disjoint
    pair among the simultaneous intervals, so a liberal p-value alone cannot
    produce a heterogeneity claim.

    A country's value may be a {template: pool values} dict: its mean is then
    the mean of its template means and its variance the stratified one
    (`_cell_moments`), with Satterthwaite df in place of n_c - 1.
    """
    rng = rng or np.random.default_rng(0)
    countries = sorted(by_country)
    cells = [_country_cells(by_country[c]) for c in countries]
    if len(cells) < 2 or any(g.size < 2 for cs in cells for g in cs):
        return {"statistic": float("nan"), "p_value": float("nan"),
                "non_estimable": True,
                "reason": "need >=2 pools in every (country, template) cell of "
                          ">=2 countries"}
    n = np.array([sum(g.size for g in cs) for cs in cells], dtype=np.float64)
    means, v, dfc = (np.array(x) for x in zip(*(_cell_moments(cs) for cs in cells)))
    if np.any(v <= 0):
        return {"statistic": float("nan"), "p_value": float("nan"),
                "non_estimable": True,
                "reason": "a country has zero pool-level variance"}

    K = len(cells)
    f_obs, df2 = _welch_f_v(means, v, dfc, K)
    p = _f_sf(f_obs, K - 1, df2)

    centred = [[g - g.mean() for g in cs] for cs in cells]
    null = np.empty(n_perm)
    for b in range(n_perm):
        sm = np.empty(K)
        sv = np.empty(K)
        for j, cs in enumerate(centred):
            smp = [g[rng.integers(0, g.size, g.size)] for g in cs]
            sm[j], sv[j], _ = _cell_moments(smp)
        sv = np.maximum(sv, 1e-300)
        null[b] = _welch_f_v(sm, sv, dfc, K)[0]
    p_boot = float((1.0 + np.sum(null >= f_obs)) / (n_perm + 1.0))

    min_pools = int(n.min())
    return {
        "statistic": float(f_obs),
        "df1": K - 1, "df2": float(df2),
        "p_value": float(p),
        "p_value_bootstrap_conservative": p_boot,
        "countries": countries,
        "country_means": {c: float(m) for c, m in zip(countries, means)},
        "country_n_pools": {c: int(x) for c, x in zip(countries, n)},
        "spread": float(means.max() - means.min()),
        "min_pools_per_country": min_pools,
        "underpowered": bool(min_pools < MIN_POOLS_FOR_HETEROGENEITY),
        "calibration_note": (
            "Welch is liberal below %d pools/country (9.2%% actual at 4, "
            "nominal 5%%); a heterogeneity claim additionally requires a "
            "disjoint pair among the simultaneous intervals."
            % MIN_POOLS_FOR_HETEROGENEITY),
        "non_estimable": False,
    }


def _cell_moments(cells):
    """(mean, variance of the mean, Satterthwaite df) of one country from its
    template cells: the mean of the template means, with variance
    `(1/T^2) sum_t s_t^2/n_t`.  A single cell gives (mean, s^2/n, n-1).

    Round-5 code review, P1: Welch and the simultaneous intervals used the
    pooled per-country vector, so a country with four surviving A pools and
    two B pools weighted template A twice -- a different country value from
    the balanced one the A1 marginal and the spread rule use."""
    T = float(len(cells))
    u = np.array([g.var(ddof=1) / g.size / (T * T) for g in cells])
    nn = np.array([g.size for g in cells], dtype=np.float64)
    tot = float(u.sum())
    den = float(np.sum(u ** 2 / (nn - 1.0)))
    return (_balanced_mean(cells), tot,
            (tot * tot / den) if den > 0 else float("inf"))


def _welch_f_v(means, v, dfc, K):
    """Welch's F from each group's variance of the mean `v` and its df."""
    w = 1.0 / v
    W = w.sum()
    gm = float((w * means).sum() / W)
    A = float((w * (means - gm) ** 2).sum() / (K - 1))
    lam = float(3.0 / (K * K - 1) * np.sum((1 - w / W) ** 2 / dfc))
    if lam <= 0:
        return A, float("inf")
    return A / (1.0 + 2.0 * (K - 2) * lam / 3.0), 1.0 / lam


def heterogeneity_verdict(het, intervals):
    """The pre-registered A4 decision rule, in one place.

    Heterogeneity is claimed only when the Welch test rejects AND at least one
    pair of simultaneous country intervals is disjoint.  The second condition
    is what keeps Welch's small-sample liberality from becoming a claim.
    """
    if het.get("non_estimable") or intervals.get("non_estimable"):
        return {"verdict": "non_estimable",
                "reason": het.get("reason") or intervals.get("reason")}
    iv = intervals["intervals"]
    disjoint = [(a, b) for i, a in enumerate(sorted(iv))
                for b in sorted(iv)[i + 1:]
                if iv[a][1] < iv[b][0] or iv[b][1] < iv[a][0]]
    rejects = het["p_value"] < 0.05
    return {
        "verdict": "heterogeneous" if (rejects and disjoint) else "not_shown",
        "welch_rejects": bool(rejects),
        "disjoint_pairs": disjoint,
        "underpowered": bool(het.get("underpowered")),
    }


def simultaneous_country_intervals(by_country, alpha=0.05, n_boot=10000,
                                   rng=None, max_degenerate_frac=0.01):
    """R2: simultaneous (not 28 pairwise) intervals on the per-country deltas.

    Bootstrap-t max-modulus: the critical value is the 1-alpha quantile of the
    maximum studentised deviation over countries.

    The hazard the code reviewer found and reproduced: with only a handful of
    pools per country, a bootstrap resample is CONSTANT with non-negligible
    probability (1/64 at n=4; 11.8% somewhere among eight countries), its
    standard error is zero, and dividing by a 1e-12 floor produced a critical
    value of 8.1e10 -- "intervals" spanning billions of log units that would
    have been reported as 95% bands.  Degenerate resamples are now discarded
    and counted, and if they are common enough to distort the quantile the
    whole result is returned `non_estimable`.  The honest fix for that case is
    more pools, not a wider band.

    Template cells as in `heterogeneity_test`: resampling is within each
    (country, template) cell and the studentisation is the stratified SE.
    """
    rng = rng or np.random.default_rng(0)
    countries = sorted(by_country)
    cells = [_country_cells(by_country[c]) for c in countries]
    if not cells or any(g.size < 2 for cs in cells for g in cs):
        return {"non_estimable": True,
                "reason": "need >=2 pools per (country, template) cell"}
    mom = [_cell_moments(cs) for cs in cells]
    if any(v <= 0 for _, v, _ in mom):
        return {"non_estimable": True,
                "reason": "a country has zero pool-level variance; interval "
                          "width is not estimable from these pools"}
    means = np.array([m for m, _, _ in mom])
    ses = np.sqrt(np.array([v for _, v, _ in mom]))

    kept = []
    degenerate = 0
    for _ in range(n_boot):
        t, ok = [], True
        for cs, m in zip(cells, means):
            sm, sv, _ = _cell_moments([g[rng.integers(0, g.size, g.size)]
                                       for g in cs])
            if sv <= 0:
                ok = False
                break
            t.append(abs(sm - m) / np.sqrt(sv))
        if ok:
            kept.append(max(t))
        else:
            degenerate += 1
    frac = degenerate / float(n_boot)
    if not kept or frac > max_degenerate_frac:
        return {"non_estimable": True,
                "reason": f"{frac:.1%} of bootstrap resamples were degenerate "
                          f"(> {max_degenerate_frac:.0%}); too few pools for a "
                          "calibrated simultaneous band",
                "degenerate_fraction": frac}
    crit = float(np.quantile(np.asarray(kept), 1 - alpha))
    return {
        "critical_value": crit,
        "degenerate_fraction": frac,
        "intervals": {c: (float(m - crit * s_), float(m + crit * s_))
                      for c, m, s_ in zip(countries, means, ses)},
        "point": {c: float(m) for c, m in zip(countries, means)},
        "non_estimable": False,
    }


#: A study with fewer pools than this cannot support an equivalence verdict.
MIN_POOLS_FOR_EQUIVALENCE = 4


def equivalence_test(deltas, margin_log=None, margin_proportional=0.10,
                     alpha=0.05, n_boot=10000, rng=None):
    """R6: a bounded result, not a non-significant p-value.

    Two corrections from the code review are built in.

    **Asymmetric margins.**  A 10% loss is `log(0.90) = -0.1054`, not
    `-log(1.10) = -0.0953`.  Using symmetric +/- log(1.10) bounds made the
    harmful boundary a 9.09% loss, so 32 pools all showing a 9.5% loss were
    declared `harmful` under a rule that was supposed to require 10%.  The
    bounds are now `[log(1 - p), log(1 + p)]`.

    **Calibrated bounds.**  The percentile bootstrap was anti-conservative at
    the pool counts this study can afford: measured 12.8% false equivalence at
    4 pools and 8.4% at 8, against a nominal 5%.  The primary decision is
    therefore TOST with Student-t bounds, which is calibrated under approximate
    normality of the pool-level contrasts; the percentile bootstrap is retained
    as a reported sensitivity, never as the decision rule.

    Verdicts: 'harmful' (the upper one-sided bound is below the loss margin),
    'equivalent' (TOST: both one-sided tests reject), 'inconclusive', or
    'non_estimable'.
    """
    rng = rng or np.random.default_rng(0)
    x = np.asarray([d for d in deltas if np.isfinite(d)], dtype=np.float64)
    p = (float(np.expm1(margin_log)) if margin_log is not None
         else float(margin_proportional))
    lower_margin = float(np.log1p(-p))            # e.g. log(0.90) = -0.10536
    upper_margin = float(np.log1p(p))             # e.g. log(1.10) = +0.09531
    margins = {"margin_log": (lower_margin, upper_margin),
               "margin_proportional": p}
    if x.size < MIN_POOLS_FOR_EQUIVALENCE:
        return {"verdict": "non_estimable", "n_pools": int(x.size), **margins,
                "reason": f"need >= {MIN_POOLS_FOR_EQUIVALENCE} estimable pools"}

    n = x.size
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    if se <= 0:
        return {"verdict": "non_estimable", "n_pools": n, "mean_delta": mean,
                **margins, "reason": "zero pool-level variance"}
    t_crit = _student_t_ppf(1 - alpha, n - 1)
    lo, hi = mean - t_crit * se, mean + t_crit * se      # the (1-2a) interval

    if hi < lower_margin:
        verdict = "harmful"
    elif lo > lower_margin and hi < upper_margin:
        verdict = "equivalent"
    else:
        verdict = "inconclusive"

    boots = np.array([x[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    b_lo, b_hi = (float(v) for v in np.quantile(boots, [alpha, 1 - alpha]))
    return {
        "verdict": verdict, "mean_delta": mean, "ci": (lo, hi),
        "se": se, "t_crit": float(t_crit), "df": int(n - 1),
        **margins,
        "bootstrap_ci_sensitivity": (b_lo, b_hi),
        "n_pools": n,
    }


#: A country with fewer pools than this contributes no variance estimate, and
#: an estimator that silently treats it as noiseless is not the estimator the
#: pre-registered margin was written about.
MIN_POOLS_PER_COUNTRY_FOR_EQUIVALENCE = 2


def _country_cells(v):
    """A country's pool values as a list of per-template arrays.  A plain
    vector is one cell; a {template: values} dict is one cell per template."""
    if isinstance(v, dict):
        return [np.asarray(v[t], dtype=np.float64) for t in sorted(v)]
    return [np.asarray(v, dtype=np.float64)]


def _balanced_mean(cells):
    """Mean of the template-cell means.  Round-4 code review: averaging a
    country's surviving pools weights its templates by how many pools of each
    survived, although the design fixes them at equal counts."""
    return float(np.mean([g.mean() for g in cells]))


def _fixed_country_moments(by_country):
    """(theta, se, df, per-country stats) for the equal-weight country marginal.

    The estimand is `theta = mean_c theta_c` with the country set FIXED by
    design.  Its sampling variance therefore comes from pools within countries
    and from nowhere else:

        Var(theta_hat) = (1/K^2) * sum_c s_c^2 / n_c
        df             = Satterthwaite over the same K terms

    Round-2 code review, P1: the runner instead handed the K country means to
    `equivalence_test`, which is an estimator for K exchangeable replicates.
    That makes the *spread between countries* the noise, although the countries
    are not a sample of anything, and the resulting test is miscalibrated in
    both directions.  The reviewer's construction -- seven country means at 0
    and an eighth at 8*log(1.1), pool SDs 0.02 except one at 0.8 -- gives
    10.13% false equivalence over 200,000 datasets at a nominal 5%, where this
    calculation gives 4.42% on the same data.  The same substitution also made
    the minimum-pool rule a minimum-*country* rule: eight countries with one
    pool each passed `MIN_POOLS_FOR_EQUIVALENCE = 4` and returned `equivalent`
    on a zero-width interval.

    A country's value may be a {template: pool values} dict, in which case the
    country mean is the mean of its template means, `s_c^2/n_c` becomes
    `(1/T^2) sum_t s_ct^2/n_ct`, and every cell needs >= 2 pools.
    """
    countries = sorted(by_country)
    cells = [_country_cells(by_country[c]) for c in countries]
    stats = {"countries": countries,
             "country_means": {}, "country_n_pools": {}}
    if len(cells) < 2:
        return None, None, None, {**stats, "reason": "need >= 2 countries"}
    means = np.array([_balanced_mean(cs) for cs in cells])
    stats["country_means"] = {c: float(m) for c, m in zip(countries, means)}
    stats["country_n_pools"] = {c: int(sum(g.size for g in cs))
                                for c, cs in zip(countries, cells)}
    thin = [c for c, cs in zip(countries, cells)
            if any(g.size < MIN_POOLS_PER_COUNTRY_FOR_EQUIVALENCE for g in cs)]
    if thin:
        return None, None, None, {
            **stats,
            "reason": "countries with < %d pools carry no variance estimate: %s"
                      % (MIN_POOLS_PER_COUNTRY_FOR_EQUIVALENCE, ", ".join(thin))}
    K = float(len(cells))
    # One variance term per (country, template) cell; a plain pool vector is a
    # single cell and reproduces the per-country formula above exactly.
    u = np.array([g.var(ddof=1) / g.size / (len(cs) ** 2) / (K * K)
                  for cs in cells for g in cs])
    n = np.array([g.size for cs in cells for g in cs], dtype=np.float64)
    total = float(u.sum())
    if not np.isfinite(total) or total <= 0:
        return None, None, None, {
            **stats,
            "reason": "every country has zero pool-level variance; the width "
                      "of an interval is not estimable from these pools"}
    denom = float(np.sum(u ** 2 / (n - 1.0)))
    df = (total ** 2 / denom) if denom > 0 else float("inf")
    return float(means.mean()), float(np.sqrt(total)), float(df), stats


def equivalence_test_country_marginal(by_country, margin_proportional=0.10,
                                      alpha=0.05, expected_countries=None):
    """TOST on the equal-weight country marginal -- the pre-registered target.

    Verdicts match `equivalence_test`: 'harmful', 'equivalent', 'inconclusive'
    or 'non_estimable'.  A country named in `expected_countries` with no
    estimable pool makes the result `non_estimable`: a marginal over the
    survivors answers a different question from the one the margin was written
    about, and reporting it under the same verdict vocabulary is how an
    incomplete design acquires a complete-design claim.
    """
    p = float(margin_proportional)
    lower_margin = float(np.log1p(-p))
    upper_margin = float(np.log1p(p))
    margins = {"margin_log": (lower_margin, upper_margin),
               "margin_proportional": p, "estimand": "equal_weight_country_marginal"}
    present = sorted(by_country)
    missing = sorted(set(expected_countries or []) - set(present))
    if missing:
        return {"verdict": "non_estimable", **margins,
                "missing_countries": missing,
                "reason": "no estimable pool for " + ", ".join(missing)}
    theta, se, df, stats = _fixed_country_moments(by_country)
    n_pools = int(sum(g.size for v in by_country.values() for g in _country_cells(v)))
    if theta is None:
        return {"verdict": "non_estimable", **margins, "missing_countries": [],
                "n_pools": n_pools, "n_countries": len(present),
                "reason": stats["reason"], **{k: stats[k] for k in
                                              ("country_means", "country_n_pools")}}
    if n_pools < MIN_POOLS_FOR_EQUIVALENCE:
        return {"verdict": "non_estimable", **margins, "missing_countries": [],
                "n_pools": n_pools, "n_countries": len(present),
                "reason": f"need >= {MIN_POOLS_FOR_EQUIVALENCE} estimable pools"}
    t_crit = _student_t_ppf(1 - alpha, df)
    lo, hi = theta - t_crit * se, theta + t_crit * se
    if hi < lower_margin:
        verdict = "harmful"
    elif lo > lower_margin and hi < upper_margin:
        verdict = "equivalent"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict, "mean_delta": theta,
        "proportional_change": float(np.expm1(theta)),
        "ci": (float(lo), float(hi)), "se": se, "df": df,
        "t_crit": float(t_crit), **margins,
        "missing_countries": [],
        "n_pools": n_pools, "n_countries": len(present),
        "country_means": stats["country_means"],
        "country_n_pools": stats["country_n_pools"],
    }


def country_marginal(by_country, alpha=0.05, n_boot=10000, rng=None,
                     expected_countries=None, expected_cells=None,
                     observed_cells=None):
    """R2/P1-5: the EQUAL-WEIGHT country marginal, with a design-preserving CI.

    Concatenating every pool and bootstrapping them as one exchangeable sample
    is wrong twice over: countries with more surviving pools get more weight
    (and non-estimable pools are not missing at random -- they are missing
    exactly where authentic artifacts are scarce), and resampling across
    countries adds variation in country composition although the eight
    countries are FIXED by design, not sampled.

    This estimates `mean_c mean_pools(delta)` with equal country weight and
    resamples pools **within** each country, holding the country set fixed.

    Two round-2 code-review findings are answered here.

    **The interval.**  The unstudentised percentile interval this function used
    to report undercovers at the pool counts the study can afford: 88.92%
    coverage at 4 pools per country and 92.38% at 8, against a nominal 95%.
    The primary interval is now the Student-t interval built from the
    fixed-country variance (`_fixed_country_moments`), which is the same
    quantity the equivalence verdict is taken on; the percentile bootstrap is
    retained as `ci_percentile`, a reported sensitivity and never the decision.

    **What "complete" means.**  `complete_design` used to look only at
    countries, so deleting every pool of one of the two templates left it
    `True` although the marginal had silently become a single-template
    marginal.  Pass `expected_cells` and `observed_cells` as (country,
    template) sets and an absent cell is named in `missing_cells` and makes
    the design incomplete.
    """
    rng = rng or np.random.default_rng(0)
    countries = sorted(by_country)
    groups = [_country_cells(by_country[c]) for c in countries]
    missing = sorted(set(expected_countries or []) - set(countries))
    missing_cells = sorted(set(expected_cells or []) - set(observed_cells or []))
    design = {"missing_countries": missing,
              "missing_cells": [list(c) for c in missing_cells],
              "complete_design": not missing and not missing_cells}
    if not groups or any(g.size == 0 for cs in groups for g in cs):
        return {"non_estimable": True, "reason": "a country has no estimable pool",
                **design}
    point = float(np.mean([_balanced_mean(cs) for cs in groups]))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        # resample pools within each (country, template) cell
        boots[b] = np.mean([np.mean([g[rng.integers(0, g.size, g.size)].mean()
                                     for g in cs]) for cs in groups])
    p_lo, p_hi = (float(v) for v in np.quantile(boots, [alpha / 2, 1 - alpha / 2]))
    theta, se, df, stats = _fixed_country_moments(by_country)
    if theta is None:
        ci, ci_method, ci_reason = None, "non_estimable", stats["reason"]
    else:
        t_crit = _student_t_ppf(1 - alpha / 2, df)
        ci = (float(theta - t_crit * se), float(theta + t_crit * se))
        ci_method, ci_reason = "student_t_fixed_countries", None
    return {
        "mean_delta": point,
        "proportional_change": float(np.expm1(point)),
        "ci": ci,
        "ci_method": ci_method,
        "ci_non_estimable_reason": ci_reason,
        "se": se,
        "df": df,
        "ci_percentile": (p_lo, p_hi),
        "country_means": {c: _balanced_mean(cs) for c, cs in zip(countries, groups)},
        "country_n_pools": {c: int(sum(g.size for g in cs))
                            for c, cs in zip(countries, groups)},
        **design,
        "non_estimable": False,
    }


def _exact_match_rarefied(labels, m, n_sub, rng):
    """`n_sub` draws of `cultural_diversity_strict(subset, batch_size=m)`, vectorised.

    Under the exact-match kernel the similarity matrix of a set is block-
    diagonal with all-ones blocks, so the eigenvalues of K/m are the label
    counts divided by m and the Vendi score is exp(H(counts/m)).  This computes
    that directly for all subsets at once.  The real run needs 10,000 null
    draws x 200 subsets per pool and endpoint; through `eigvalsh` that was
    about a CPU-day, and this is the same number (pinned against the
    eigendecomposition by `test_vectorised_rarefaction_matches_the_vendi_path`).
    """
    _, codes = np.unique(np.asarray([str(x) for x in labels], dtype=object),
                         return_inverse=True)
    L, u = len(labels), int(codes.max()) + 1
    pick = np.argsort(rng.random((n_sub, L)), axis=1)[:, :m]
    counts = np.zeros((n_sub, u))
    np.add.at(counts, (np.repeat(np.arange(n_sub), m), codes[pick].ravel()), 1.0)
    p = counts / m
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    return np.exp(h) / m


def rarefy_to_common_count(labels, m, n_sub=200, rng=None):
    """R3: evaluate the authenticity-conditional endpoint at a fixed count.

    Filtering to authentic samples leaves a *condition-dependent* count, so
    empirical entropy moves with sample size even at unchanged underlying
    diversity -- matching k does not match the number of authentic
    observations.  Sets with fewer than `m` authentic samples are NON-ESTIMABLE
    and must be reported as such together with their authenticity deficit, not
    silently dropped.
    """
    labels = list(labels)
    if len(labels) < m:
        return {"value": float("nan"), "non_estimable": True,
                "n_authentic": len(labels), "m": int(m)}
    rng = rng or np.random.default_rng(0)
    if len(labels) == m:
        vals = [cultural_diversity_strict(labels, batch_size=m)]
    else:
        vals = _exact_match_rarefied(labels, m, n_sub, rng)
    return {"value": float(np.mean(vals)), "non_estimable": False,
            "n_authentic": len(labels), "m": int(m)}
