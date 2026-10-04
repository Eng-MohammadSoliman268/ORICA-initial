import numpy as np

from orica.controllers import CoolingController
from orica.core import ORICAConfig, OnlineORICA


def test_cooling_reference_values():
    c = CoolingController(lambda0=0.995, gamma=0.60)
    idx = np.arange(1, 9, dtype=np.float64)

    white = float(c.whitening_lambda(idx, np))
    ica = np.asarray(c.ica_lambdas(idx, 0.0, np))

    assert np.isclose(white, 0.995 / (4.0**0.60), rtol=0, atol=1e-15)
    assert np.isclose(ica[-1], 0.995 / (8.0**0.60), rtol=0, atol=1e-15)
    assert np.all((ica > 0) & (ica < 1))


def test_core_runs_and_preserves_orthogonality():
    rng = np.random.default_rng(1234)
    D = 4
    block = 8

    config = ORICAConfig(
        n_channels=D,
        sfreq=128.0,
        block_size=block,
        backend="numpy",
        dtype="float64",
        nsi_tau_s=0.31193161193957875,
        orthogonalize_every=1,
    )
    model = OnlineORICA(config, CoolingController())

    X = rng.laplace(size=(D, 256)).astype(np.float64)
    for start in range(0, X.shape[1], block):
        out = model.update(X[:, start : start + block], return_cpu=True)
        assert out["Y"].shape == (D, block)
        assert np.all(np.isfinite(out["Y"]))

    state = model.get_state(cpu=True)
    W = state["W"]
    B = model.get_unmixing(cpu=True)

    assert state["samples_seen"] == X.shape[1]
    assert np.all(np.isfinite(state["M"]))
    assert np.all(np.isfinite(W))
    assert np.allclose(B, W @ state["M"], rtol=0, atol=1e-13)
    assert np.max(np.abs(W @ W.T - np.eye(D))) < 1e-10


def test_transform_does_not_change_state():
    rng = np.random.default_rng(7)
    D = 3
    model = OnlineORICA(
        ORICAConfig(n_channels=D, sfreq=128.0, block_size=8, backend="numpy"),
        CoolingController(),
    )

    X = rng.normal(size=(D, 8))
    model.update(X)
    before = model.get_state(cpu=True)
    Y = model.transform(X, cpu=True)
    after = model.get_state(cpu=True)

    assert Y.shape == X.shape
    assert np.array_equal(before["M"], after["M"])
    assert np.array_equal(before["W"], after["W"])
    assert before["samples_seen"] == after["samples_seen"]


def test_experimental_v43_equals_cooling_when_boost_disabled():
    from orica.experimental import GeneralSelfTuningControllerV43

    rng = np.random.default_rng(55)
    D = 4
    fs = 128.0
    block = 8

    cfg_a = ORICAConfig(D, fs, block, backend="numpy")
    cfg_b = ORICAConfig(D, fs, block, backend="numpy")

    cooling = OnlineORICA(cfg_a, CoolingController())
    v43 = OnlineORICA(
        cfg_b,
        GeneralSelfTuningControllerV43(
            D,
            fs,
            block,
            protect_min_s=1e9,
            protect_max_s=1e9,
        ),
    )

    for _ in range(10):
        X = rng.laplace(size=(D, block))
        cooling.update(X)
        v43.update(X)

    s_c = cooling.get_state(cpu=True)
    s_v = v43.get_state(cpu=True)

    assert np.array_equal(s_c["M"], s_v["M"])
    assert np.array_equal(s_c["W"], s_v["W"])
    assert np.array_equal(s_c["Rn"], s_v["Rn"])
    assert np.array_equal(
        cooling.get_unmixing(cpu=True),
        v43.get_unmixing(cpu=True),
    )
