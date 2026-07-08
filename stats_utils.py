"""Paired stats dung chung (khong phu thuoc torch/vision)."""
import math


def mean_se(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var / n)


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _t_sf(t, df):
    if df <= 0 or not math.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    ) if 0.0 < x < 1.0 else 0.0

    def _betacf(x, a, b):
        MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c = 1.0
        d = 1.0 - qab * x / qap
        if abs(d) < FPMIN: d = FPMIN
        d = 1.0 / d; h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN: d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN: c = FPMIN
            d = 1.0 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < FPMIN: d = FPMIN
            c = 1.0 + aa / c
            if abs(c) < FPMIN: c = FPMIN
            d = 1.0 / d; de = d * c; h *= de
            if abs(de - 1.0) < EPS: break
        return h
    if x < (a + 1.0) / (a + b + 2.0):
        ix = bt * _betacf(x, a, b) / a
    else:
        ix = 1.0 - bt * _betacf(1.0 - x, b, a) / b
    return max(0.0, min(1.0, ix))


def paired_t(a, b):
    d = [ai - bi for ai, bi in zip(a, b)]
    n = len(d)
    if n < 2:
        return (float("nan"), float("nan"), float("nan"))
    md = sum(d) / n
    var = sum((x - md) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0.0:
        return (md, float("inf") if md != 0 else 0.0, 0.0 if md != 0 else 1.0)
    t = md / (sd / math.sqrt(n))
    return (md, t, _t_sf(t, n - 1))


def wilcoxon(a, b):
    d = [ai - bi for ai, bi in zip(a, b) if ai - bi != 0.0]
    n = len(d)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    Wp = sum(ranks[i] for i in range(n) if d[i] > 0)
    Wm = sum(ranks[i] for i in range(n) if d[i] < 0)
    W = min(Wp, Wm)
    mu = n * (n + 1) / 4.0
    from collections import Counter
    tie_counts = Counter(abs(x) for x in d)
    tie_term = sum(t ** 3 - t for t in tie_counts.values())
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if sigma2 <= 0:
        return (W, float("nan"), float("nan"))
    z = (W - mu + 0.5 * (1 if W < mu else -1)) / math.sqrt(sigma2)
    p = 2.0 * _norm_cdf(-abs(z))
    return (W, z, p)


