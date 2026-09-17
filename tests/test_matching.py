import pandas as pd
import pytest

from matching import ObservationMatcher  # type: ignore


def test_exact_date_matching():
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-10-23",
                    "2018-10-31",
                    "2018-11-08",
                ]
            ),
            "value": [0.318, 0.305, 0.291],
        }
    )

    simulations = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-10-23",
                    "2018-10-24",
                    "2018-10-31",
                    "2018-11-01",
                    "2018-11-08",
                ]
            ),
            "value": [0.319, 0.316, 0.315, 0.300, 0.294],
        }
    )

    matcher = ObservationMatcher()
    result = matcher.match(observations, simulations)

    assert len(result) == 3

    assert list(result.columns) == [
        "date",
        "observed",
        "simulated",
    ]

    assert list(result["date"]) == [
        pd.Timestamp("2018-10-23"),
        pd.Timestamp("2018-10-31"),
        pd.Timestamp("2018-11-08"),
    ]

    assert list(result["observed"]) == [
        pytest.approx(0.318),
        pytest.approx(0.305),
        pytest.approx(0.291),
    ]

    assert list(result["simulated"]) == [
        pytest.approx(0.319),
        pytest.approx(0.315),
        pytest.approx(0.294),
    ]


def test_non_observed_simulation_dates_are_not_used():
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-10-23", "2018-10-31"]),
            "value": [0.318, 0.305],
        }
    )

    simulations = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-10-23",
                    "2018-10-24",
                    "2018-10-25",
                    "2018-10-31",
                ]
            ),
            "value": [0.319, 0.316, 0.310, 0.315],
        }
    )

    result = ObservationMatcher().match(
        observations,
        simulations,
    )

    assert len(result) == 2
    assert "2018-10-24" not in result["date"].astype(str).tolist()
    assert "2018-10-25" not in result["date"].astype(str).tolist()


def test_observation_without_simulation_is_not_used():
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-10-23",
                    "2018-10-31",
                    "2018-11-08",
                ]
            ),
            "value": [0.318, 0.305, 0.291],
        }
    )

    simulations = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-10-23",
                    "2018-11-08",
                ]
            ),
            "value": [0.319, 0.294],
        }
    )

    result = ObservationMatcher().match(
        observations,
        simulations,
    )

    assert len(result) == 2

    assert list(result["date"]) == [
        pd.Timestamp("2018-10-23"),
        pd.Timestamp("2018-11-08"),
    ]


def test_no_date_shift_or_interpolation():
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-10-23"]),
            "value": [0.318],
        }
    )

    simulations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-10-24"]),
            "value": [0.319],
        }
    )

    result = ObservationMatcher().match(
        observations,
        simulations,
    )

    assert result.empty


def test_duplicate_observation_dates_are_rejected():
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-10-23",
                    "2018-10-23",
                ]
            ),
            "value": [0.318, 0.320],
        }
    )

    simulations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-10-23"]),
            "value": [0.319],
        }
    )

    with pytest.raises(
        ValueError,
        match="observations contains duplicate dates",
    ):
        ObservationMatcher().match(
            observations,
            simulations,
        )


def test_duplicate_simulation_dates_are_rejected():
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-10-23"]),
            "value": [0.318],
        }
    )

    simulations = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2018-10-23",
                    "2018-10-23",
                ]
            ),
            "value": [0.319, 0.320],
        }
    )

    with pytest.raises(
        ValueError,
        match="simulations contains duplicate dates",
    ):
        ObservationMatcher().match(
            observations,
            simulations,
        )


def test_missing_required_column_is_rejected():
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-10-23"]),
            "observed": [0.318],
        }
    )

    simulations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-10-23"]),
            "value": [0.319],
        }
    )

    with pytest.raises(
        ValueError,
        match="observations is missing required columns",
    ):
        ObservationMatcher().match(
            observations,
            simulations,
        )
