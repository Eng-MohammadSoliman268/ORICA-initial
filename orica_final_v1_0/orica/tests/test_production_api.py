import numpy as np

from orica.api import (
    ORICA_FINAL_VERSION,
    create_final_orica,
    process_orica_block,
    run_orica_array,
)


def test_create_final_orica_profile():
    model, info = create_final_orica(
        n_channels=4,
        sfreq=500.0,
        block_size=8,
        backend="numpy",
    )

    assert info["version"] == ORICA_FINAL_VERSION
    assert info["controller"] == "CoolingController"
    assert info["lambda0"] == 0.995
    assert info["gamma"] == 0.60
    assert model.backend == "numpy"


def test_process_orica_block_and_diagnostics():
    rng = np.random.default_rng(99)
    model, _ = create_final_orica(4, 500.0, backend="numpy")
    X = rng.normal(size=(4, 8))

    Y, diag = process_orica_block(
        model,
        X,
        return_diagnostics=True,
        output_mode="updated",
    )

    assert Y.shape == X.shape
    assert np.all(np.isfinite(Y))
    assert diag["all_finite"] is True
    assert diag["samples_seen"] == 8
    assert diag["blocks_seen"] == 1
    assert diag["orthogonality_error"] < 1e-10


def test_offline_runner_keeps_shape():
    rng = np.random.default_rng(2026)
    X = rng.laplace(size=(3, 37))

    Y, model, info = run_orica_array(
        X,
        sfreq=128.0,
        block_size=8,
        backend="numpy",
        keep_tail=True,
    )

    assert Y.shape == X.shape
    assert model.samples_seen == X.shape[1]
    assert info["version"] == ORICA_FINAL_VERSION
    assert np.all(np.isfinite(Y))
