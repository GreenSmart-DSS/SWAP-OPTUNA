import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from run_depth_diagnostics import main


def test_depth_diagnostics_creates_layer_csvs(tmp_path, monkeypatch):
    """Create one simulation CSV for every requested soil layer."""

    source_vap = PROJECT_DIR / "model" / "results" / "result.vap"

    observation_path = PROJECT_DIR / "model" / "observation.csv"

    test_project = tmp_path / "project"
    test_model = test_project / "model"
    test_results = test_model / "results"

    test_results.mkdir(parents=True)

    test_vap = test_results / "result.vap"
    test_vap.write_bytes(source_vap.read_bytes())

    test_observation = test_model / "observation.csv"
    test_observation.write_bytes(observation_path.read_bytes())

    monkeypatch.setattr(
        "run_depth_diagnostics.PROJECT_DIR",
        test_project,
    )

    main()

    output_dir = test_project / "depth_diagnostics"

    expected_layers = [
        "layer_0_5",
        "layer_5_10",
        "layer_10_20",
        "layer_20_30",
        "layer_5_60",
    ]

    for layer_name in expected_layers:
        csv_path = output_dir / layer_name / "simulation.csv"

        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

        data = pd.read_csv(csv_path)

        assert not data.empty
        assert list(data.columns) == [
            "date",
            "wcontent",
            "phead",
            "hconduc",
            "temp",
        ]
