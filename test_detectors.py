"""Unit tests for the detectors: known input -> known statistic, alarms,
restarts, finite-time EWMA limits, and the CUSUM <-> SPRT link.
Run:  python -m pytest -q test_detectors.py"""
import math

import numpy as np
import pytest

from detectors import CUSUM, EWMA, MEWMA, SPRT, Shewhart, ewma_limit, make


# ---- common interface ------------------------------------------------------
@pytest.mark.parametrize("method", ["shewhart", "sprt", "cusum", "ewma"])
def test_interface_scalar(method):
    d = make(method, mu0=10.0, sigma0=2.0)
    out = d.update(10.0)
    assert set(out) >= {"alarm", "side"}
    assert isinstance(out["alarm"], bool) and out["side"] is None
    assert any(k.startswith("statistic") for k in out)
    assert any(k.startswith("threshold") for k in out)
    assert isinstance(d.params, dict)
    with pytest.raises(ValueError):
        d.set_params(nonsense=1)


def test_interface_mewma():
    d = MEWMA(mu0=[0, 0], sigma0=[1, 1], cov=np.eye(2), lam=0.2, h=10)
    out = d.update([0.0, 0.0])
    assert out["alarm"] is False and out["side"] is None and out["statistic"] == 0.0


@pytest.mark.parametrize("method,bad", [("cusum", {"h": -1}), ("ewma", {"lam": 0}), ("ewma", {"lam": 1.5}),
                                         ("sprt", {"alpha": 0.6, "beta": 0.5}), ("shewhart", {"k": 0}),
                                         ("mewma", {"h": 0}), ("cusum", {"restart": "maybe"})])
def test_invalid_params(method, bad):
    with pytest.raises(ValueError):
        make(method, **bad)


def test_vectorised_equals_loop():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 4))
    x[150:, 2] += 1.5
    vec = CUSUM(kappa=0.5, h=4)
    one = [CUSUM(kappa=0.5, h=4) for _ in range(4)]
    for row in x:
        v = vec.update(row)
        for j in range(4):
            o = one[j].update(row[j])
            assert o["alarm"] == bool(v["alarm"][j])
            assert o["statistic_up"] == pytest.approx(v["statistic_up"][j])


# ---- Shewhart --------------------------------------------------------------
def test_shewhart():
    d = Shewhart(mu0=100, sigma0=2, k=3)
    assert d.update(105)["alarm"] is False          # 2.5 sigma
    out = d.update(107)                             # 3.5 sigma
    assert out["alarm"] and out["side"] == "up" and out["statistic"] == pytest.approx(3.5)
    assert d.update(93)["side"] == "down"


# ---- CUSUM -----------------------------------------------------------------
def test_cusum_known_values():
    d = CUSUM(mu0=0, sigma0=1, kappa=0.5, h=2)
    seq = [1.0, 1.0, -0.2, 1.5]
    # C+: 0.5, 1.0, 0.3, 1.3     C-: 0, 0, 0, 0
    exp = [0.5, 1.0, 0.3, 1.3]
    for x, e in zip(seq, exp):
        out = d.update(x)
        assert out["statistic_up"] == pytest.approx(e)
        assert out["statistic_down"] == 0.0 and not out["alarm"]
    out = d.update(1.5)                             # 1.3 + 1.0 = 2.3 > 2
    assert out["alarm"] and out["side"] == "up"
    assert d.update(0.0)["statistic_up"] == 0.0     # was reset after the alarm


def test_cusum_raw_units():
    # slide form: C+ = max(0, C+ + Y - mu0 - kappa_raw); our kappa is in sigma0 units
    d = CUSUM(mu0=50, sigma0=4, kappa=0.5, h=10)
    out = d.update(58)                              # (8 - 2) / 4 = 1.5 sigma
    assert out["statistic_up"] == pytest.approx(1.5)


def test_cusum_down_side():
    d = CUSUM(kappa=0.5, h=1)
    d.update(-1.2)
    out = d.update(-1.2)                            # C- = 0.7 + 0.7 = 1.4 > 1
    assert out["alarm"] and out["side"] == "down"


def test_cusum_continue_policy_keeps_state():
    d = CUSUM(kappa=0.0, h=1, restart="continue")
    d.update(2.0)
    assert d.update(0.0)["statistic_up"] == pytest.approx(2.0)


def test_cusum_is_repeated_sprt_reset_at_zero():
    """C+ * Delta equals the SPRT log-likelihood sum when the lower boundary
    is a reflecting barrier at 0 (log B -> 0) and kappa = Delta / 2."""
    rng = np.random.default_rng(3)
    delta = 1.2
    c = CUSUM(kappa=delta / 2, h=1e9)
    s = 0.0
    for x in rng.normal(0.3, 1, 500):
        s = max(0.0, s + delta * (x - delta / 2))
        assert c.update(x)["statistic_up"] * delta == pytest.approx(s)


# ---- SPRT ------------------------------------------------------------------
def test_sprt_llr_and_limits():
    d = SPRT(mu0=0, sigma0=1, delta=1.0, alpha=0.05, beta=0.2)
    assert d.log_a == pytest.approx(math.log(0.8 / 0.05))
    assert d.log_b == pytest.approx(math.log(0.2 / 0.95))
    out = d.update(2.0)                             # l = 1 * (2 - 0.5) = 1.5
    assert out["statistic_up"] == pytest.approx(1.5)
    assert out["statistic_down"] == pytest.approx(1 * (-2 - 0.5))   # -2.5 <= log B
    assert out["accept_h0"]
    assert d.s_down == 0.0                          # quiet restart of the down test


def test_sprt_gaussian_llr_matches_density_ratio():
    mu0, s0, delta_sig = 5.0, 2.0, 1.5
    d = SPRT(mu0=mu0, sigma0=s0, delta=delta_sig, alpha=1e-6, beta=0.4)
    x = 7.3
    f0 = math.exp(-((x - mu0) / s0) ** 2 / 2)
    f1 = math.exp(-((x - mu0 - delta_sig * s0) / s0) ** 2 / 2)
    assert d.update(x)["statistic_up"] == pytest.approx(math.log(f1 / f0))


def test_sprt_alarm_and_restart():
    d = SPRT(delta=2.0, alpha=0.01, beta=0.1)      # log A = log(90) = 4.50
    d.update(2.0)                                   # S = 2
    out = d.update(2.5)                             # S = 2 + 3 = 5 >= 4.5
    assert out["alarm"] and out["side"] == "up"
    assert d.s_up == 0.0


def test_sprt_wald_error_rate_roughly():
    """One-shot SPRT under H0: P(decide H1) should be about alpha (Wald is approximate)."""
    rng = np.random.default_rng(0)
    d = SPRT(delta=1.0, alpha=0.05, beta=0.05)
    n = 4000
    s = np.zeros(n)
    decided = np.zeros(n, dtype=int)                # 0 running, 1 H1, -1 H0
    while (decided == 0).any():
        run = decided == 0
        s[run] += 1.0 * (rng.normal(size=run.sum()) - 0.5)
        decided[run & (s >= d.log_a)] = 1
        decided[run & (s <= d.log_b)] = -1
    assert (decided == 1).mean() < 0.05             # Wald's bound: actual alpha <= nominal


# ---- EWMA ------------------------------------------------------------------
def test_ewma_known_values_and_finite_limits():
    lam, L = 0.2, 3.0
    d = EWMA(mu0=0, sigma0=1, lam=lam, L=L)
    z = 0.0
    for t, x in enumerate([1.0, -0.5, 0.25, 2.0], start=1):
        z = lam * x + (1 - lam) * z
        out = d.update(x)
        assert out["statistic"] == pytest.approx(z)
        assert out["threshold_up"] == pytest.approx(ewma_limit(lam, L, t))
        assert out["threshold_down"] == pytest.approx(-ewma_limit(lam, L, t))
    # first limit = L * lambda, steady state = L * sqrt(lam / (2 - lam))
    assert ewma_limit(lam, L, 1) == pytest.approx(L * lam)
    assert ewma_limit(lam, L, 10_000) == pytest.approx(L * math.sqrt(lam / (2 - lam)))


def test_ewma_alarm_and_restart():
    d = EWMA(lam=0.5, L=1.0)
    out = d.update(3.0)                             # Z = 1.5 > 1 * 0.5
    assert out["alarm"] and out["side"] == "up"
    out = d.update(0.0)                             # restarted: Z = 0, limit back to t=1
    assert out["statistic"] == 0.0 and out["threshold_up"] == pytest.approx(0.5)


def test_ewma_irregular_time():
    rho = 0.3
    d = EWMA(lam=0.1, L=3, rho=rho)
    d.update(1.0, t=0.0)                            # first sample: dt = 1
    lam1 = 1 - math.exp(-rho * 1.0)
    out = d.update(1.0, t=4.0)                      # dt = 4: bigger weight on the new sample
    lam2 = 1 - math.exp(-rho * 4.0)
    z = lam2 * 1.0 + (1 - lam2) * lam1
    assert out["statistic"] == pytest.approx(z)


def test_ewma_lambda_one_is_shewhart():
    e, s = EWMA(lam=1.0, L=3.0), Shewhart(k=3.0)
    for x in np.random.default_rng(1).normal(0, 1.5, 300):
        assert e.update(x)["alarm"] == s.update(x)["alarm"]


# ---- MEWMA -----------------------------------------------------------------
def test_mewma_known_value():
    lam = 0.25
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    d = MEWMA(mu0=[0, 0], sigma0=[1, 1], cov=cov, lam=lam, h=100)
    x = np.array([1.0, -1.0])
    out = d.update(x)
    z = lam * x
    sz = lam / (2 - lam) * (1 - (1 - lam) ** 2) * cov
    assert out["statistic"] == pytest.approx(z @ np.linalg.inv(sz) @ z)


def test_mewma_one_sensor_is_ewma_squared():
    lam, L = 0.1, 2.7
    m, e = MEWMA(mu0=[0], sigma0=[1], cov=[[1.0]], lam=lam, h=L ** 2), EWMA(lam=lam, L=L)
    for x in np.random.default_rng(2).normal(0.2, 1, 400):
        a, b = m.update([x]), e.update(x)
        assert a["alarm"] == b["alarm"]


def test_mewma_alarm_and_reset_vectorised():
    d = MEWMA(cov=np.eye(3), lam=0.5, h=5)
    x = np.zeros((4, 3))
    x[2] = [4, 4, 4]
    out = d.update(x)
    assert out["alarm"].tolist() == [False, False, True, False]
    assert np.all(d.z[2] == 0) and d.n[2] == 0 and d.n[0] == 1


def test_refused_set_params_leaves_detector_unchanged():
    d = SPRT(alpha=0.01, beta=0.1)
    with pytest.raises(ValueError):
        d.set_params(alpha=0.95)                    # alpha + beta >= 1
    assert d.params["alpha"] == 0.01 and d.log_a == pytest.approx(math.log(0.9 / 0.01))
