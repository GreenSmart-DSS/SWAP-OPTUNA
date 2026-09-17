import math

import pandas as pd
import pytest

from metrics import MetricsCalculator  # type: ignore


def test_metrics_are_calculated_correctly():
    matched = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-10-23",
                    "2018-10-31",
                    "2018-11-08",
                ]
            ),
            "observed": [0.30, 0.35, 0.40],
            "simulated": [0.32, 0.33, 0.42],
        }
    )

    metrics = MetricsCalculator().calculate(matched)

    errors = [0.02, -0.02, 0.02]

    expected_rmse = math.sqrt(sum(error**2 for error in errors) / 3)

    expected_mae = sum(abs(error) for error in errors) / 3

    expected_mbe = sum(errors) / 3

    assert metrics.rmse == pytest.approx(expected_rmse)
    assert metrics.mae == pytest.approx(expected_mae)
    assert metrics.mbe == pytest.approx(expected_mbe)
    assert metrics.n == 3


def test_perfect_simulation_has_zero_error():
    matched = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-10-23",
                    "2018-10-31",
                    "2018-11-08",
                ]
            ),
            "observed": [0.30, 0.35, 0.40],
            "simulated": [0.30, 0.35, 0.40],
        }
    )

    metrics = MetricsCalculator().calculate(matched)

    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.mbe == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.n == 3


def test_r2_is_calculated():
    matched = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-10-23",
                    "2018-10-31",
                    "2018-11-08",
                    "2018-11-21",
                ]
            ),
            "observed": [1.0, 2.0, 3.0, 4.0],
            "simulated": [1.1, 1.9, 3.2, 3.8],
        }
    )

    metrics = MetricsCalculator().calculate(matched)

    assert metrics.r2 == pytest.approx(
        0.98,
        abs=0.01,
    )


def test_positive_mbe_indicates_overestimation():
    matched = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-10-23",
                    "2018-10-31",
                ]
            ),
            "observed": [0.30, 0.30],
            "simulated": [0.32, 0.34],
        }
    )

    metrics = MetricsCalculator().calculate(matched)

    assert metrics.mbe > 0


def test_negative_mbe_indicates_underestimation():
    matched = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-10-23",
                    "2018-10-31",
                ]
            ),
            "observed": [0.30, 0.30],
            "simulated": [0.28, 0.26],
        }
    )

    metrics = MetricsCalculator().calculate(matched)

    assert metrics.mbe < 0


def test_empty_data_is_rejected():
    matched = pd.DataFrame(
        columns=[
            "date",
            "observed",
            "simulated",
        ]
    )

    with pytest.raises(
        ValueError,
        match="empty matched data",
    ):
        MetricsCalculator().calculate(matched)


def test_missing_column_is_rejected():
    matched = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-10-23"]),
            "observed": [0.30],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        MetricsCalculator().calculate(matched)
