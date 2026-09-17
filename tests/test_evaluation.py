from pathlib import Path

import pandas as pd
import pytest

from evaluation import OutputEvaluator  # type: ignore
from vap import RequestedColumn, RequestedLayer  # type: ignore


def write_test_vap(path: Path) -> None:
    """Write a minimal VAP file containing two soil layers."""

    content = """\
date,depth,wcontent,phead,hconduc,drainage,rootext,waterflux,temp,solute1,solute2,soluteflux,top,bottom,day,dcum
cm,cm3/cm3,cm,cm/d,cm/d,cm/d,cm/d,cm/d,C,-,g/l,cm/d,cm,cm,d,d
10-Apr-2019,-2.5,0.30,-100,1,0,0,0,20,0,0,0,0,-5,100,100
10-Apr-2019,-7.5,0.25,-120,1,0,0,0,20,0,0,0,-5,-10,100,100
21-Apr-2019,-2.5,0.32,-100,1,0,0,0,20,0,0,0,0,-5,111,111
21-Apr-2019,-7.5,0.27,-120,1,0,0,0,20,0,0,0,-5,-10,111,111
26-Apr-2019,-2.5,0.31,-100,1,0,0,0,20,0,0,0,0,-5,116,116
26-Apr-2019,-7.5,0.26,-120,1,0,0,0,20,0,0,0,-5,-10,116,116
"""

    path.write_text(content, encoding="latin-1")


def write_observation_file(
    path: Path,
    rows: list[tuple[str, float]],
) -> None:
    """Write a minimal observation file."""

    data = pd.DataFrame(
        rows,
        columns=["date", "value"],
    )

    data.to_csv(path, index=False)


def test_evaluate_multiple_layers(tmp_path):
    """Evaluate multiple requested layers from one VAP file."""

    vap_path = tmp_path / "result.vap"

    write_test_vap(vap_path)

    evaluator = OutputEvaluator()

    layers = [
        RequestedLayer(0, 5),
        RequestedLayer(5, 10),
    ]

    results = evaluator.evaluate(
        vap_path=vap_path,
        layers=layers,
    )

    assert len(results) == 2

    assert results[0].layer.name == "0-5"
    assert results[1].layer.name == "5-10"

    assert len(results[0].simulation) == 3
    assert len(results[1].simulation) == 3


def test_layer_with_observations_gets_metrics(tmp_path):
    """Calculate metrics when exact observations are available."""

    vap_path = tmp_path / "result.vap"
    observation_path = tmp_path / "observation_0_5.csv"

    write_test_vap(vap_path)

    write_observation_file(
        observation_path,
        [
            ("2019-04-10", 0.29),
            ("2019-04-21", 0.31),
            ("2019-04-26", 0.30),
        ],
    )

    evaluator = OutputEvaluator()

    results = evaluator.evaluate(
        vap_path=vap_path,
        layers=[
            RequestedLayer(0, 5),
        ],
        observation_paths={
            "0-5": observation_path,
        },
    )

    result = results[0]

    assert result.has_observations
    assert result.has_metrics

    assert len(result.observations) == 3
    assert len(result.matched) == 3
    assert result.metrics is not None
    assert result.metrics.n == 3


def test_layer_without_observations_returns_simulation_only(tmp_path):
    """Return simulation without metrics when observations are absent."""

    vap_path = tmp_path / "result.vap"

    write_test_vap(vap_path)

    evaluator = OutputEvaluator()

    results = evaluator.evaluate(
        vap_path=vap_path,
        layers=[
            RequestedLayer(0, 5),
            RequestedLayer(5, 10),
        ],
    )

    assert len(results) == 2

    for result in results:
        assert result.has_observations is False
        assert result.has_metrics is False
        assert result.observations is None
        assert result.matched is None
        assert result.metrics is None
        assert len(result.simulation) == 3


def test_non_matching_dates_produce_no_metrics(tmp_path):
    """Do not calculate metrics when no dates match exactly."""

    vap_path = tmp_path / "result.vap"
    observation_path = tmp_path / "observation_0_5.csv"

    write_test_vap(vap_path)

    write_observation_file(
        observation_path,
        [
            ("2019-04-11", 0.29),
            ("2019-04-22", 0.31),
        ],
    )

    evaluator = OutputEvaluator()

    results = evaluator.evaluate(
        vap_path=vap_path,
        layers=[
            RequestedLayer(0, 5),
        ],
        observation_paths={
            "0-5": observation_path,
        },
    )

    result = results[0]

    assert result.has_observations
    assert result.matched.empty
    assert result.metrics is None


def test_invalid_observation_path_raises(tmp_path):
    """Raise an error when an observation file does not exist."""

    vap_path = tmp_path / "result.vap"
    observation_path = tmp_path / "missing.csv"

    write_test_vap(vap_path)

    evaluator = OutputEvaluator()

    with pytest.raises(FileNotFoundError):
        evaluator.evaluate(
            vap_path=vap_path,
            layers=[
                RequestedLayer(0, 5),
            ],
            observation_paths={
                "0-5": observation_path,
            },
        )


def test_evaluate_multiple_columns(tmp_path):
    """Extract multiple requested variables for multiple soil layers."""

    vap_path = tmp_path / "result.vap"
    write_test_vap(vap_path)

    evaluator = OutputEvaluator()

    layers = [
        RequestedLayer(0, 5),
        RequestedLayer(5, 10),
    ]

    columns = [
        RequestedColumn("wcontent", "weighted_mean"),
        RequestedColumn("phead", "weighted_mean"),
        RequestedColumn("hconduc", "weighted_mean"),
        RequestedColumn("temp", "weighted_mean"),
    ]

    results = evaluator.evaluate(
        vap_path=vap_path,
        layers=layers,
        columns=columns,
    )

    assert len(results) == 2

    expected_columns = [
        "date",
        "wcontent",
        "phead",
        "hconduc",
        "temp",
    ]

    for result in results:
        assert list(result.simulation.columns) == expected_columns
        assert len(result.simulation) == 3
