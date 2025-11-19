import math
from pathlib import Path

import numpy as np
import pytest

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import DynamicPBPKGNN, DynamicPBPKSimulator

CHECKPOINT = Path("models/dynamic_gnn_full/best_model.pt")


@pytest.mark.skipif(
    not CHECKPOINT.exists(),
    reason="Checkpoint treinado ausente; regressão desativada.",
)
def test_dynamic_gnn_checkpoint_regression():
    model = DynamicPBPKGNN(num_temporal_steps=100, dt=0.1)
    simulator = DynamicPBPKSimulator(
        model=model,
        device="cpu",
        checkpoint_path=str(CHECKPOINT),
        strict=True,
    )

    results = simulator.simulate(
        dose=100.0,
        clearance_hepatic=12.0,
        clearance_renal=6.5,
        partition_coeffs={"liver": 1.8, "kidney": 1.5, "brain": 0.8},
    )

    time = results["time"]
    blood = results["blood"]

    assert len(time) == 101
    assert len(blood) == 101

    cmax = float(np.max(blood))
    t_cmax = float(time[np.argmax(blood)])
    final_blood = float(blood[-1])

    assert math.isclose(cmax, 20.0, rel_tol=1e-3)
    assert math.isclose(t_cmax, 0.0, abs_tol=1e-6)
    assert math.isclose(final_blood, 0.3175, rel_tol=0.2)

    expected_endpoints = {
        "bone": 0.4362,
        "pancreas": 0.4362,
        "skin": 0.4361,
        "adipose": 0.4360,
    }

    for organ, expected in expected_endpoints.items():
        final_value = float(results[organ][-1])
        assert math.isclose(final_value, expected, rel_tol=0.2)
