import pandas as pd

from metrics import MetricsCalculator  # type: ignore
from plotting import (  # type: ignore
    plot_layer_scatter,
    plot_layer_time_series,
    plot_scatter,
    plot_time_series,
)
from vap import RequestedLayer  # type: ignore


def create_test_data():
    """Create a small matched dataset for plotting tests."""

    dates = pd.to_datetime(
        [
            "2019-04-10",
            "2019-04-21",
            "2019-04-26",
        ]
    )

    observations = pd.DataFrame(
        {
            "date": dates,
            "value": [0.3176, 0.2992, 0.3140],
        }
    )

    simulation = pd.DataFrame(
        {
            "date": dates,
            "value": [0.2684, 0.2678, 0.2784],
        }
    )

    matched = pd.DataFrame(
        {
            "date": dates,
            "observed": observations["value"],
            "simulated": simulation["value"],
        }
    )

    return observations, simulation, matched


def test_plot_time_series_creates_file(tmp_path):
    observations, simulation, matched = create_test_data()

    metrics = MetricsCalculator().calculate(matched)

    output_path = tmp_path / "time_series.png"

    plot_time_series(
        simulation=simulation,
        observations=observations,
        metrics=metrics,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_scatter_creates_file(tmp_path):
    _, _, matched = create_test_data()

    metrics = MetricsCalculator().calculate(matched)

    output_path = tmp_path / "scatter.png"

    plot_scatter(
        matched=matched,
        metrics=metrics,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_layer_time_series_with_observations(tmp_path):
    """Create a multi-layer time-series plot with observations."""

    observations, simulation, matched = create_test_data()

    metrics = MetricsCalculator().calculate(matched)

    output_path = tmp_path / "layer_0_5_time_series.png"

    plot_layer_time_series(
        simulation=simulation,
        observations=observations,
        metrics=metrics,
        layer=RequestedLayer(0, 5),
        output_path=output_path,
        title_prefix="Best Trial",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_layer_time_series_without_observations(tmp_path):
    """Create a simulation-only plot when observations are unavailable."""

    _, simulation, _ = create_test_data()

    output_path = tmp_path / "layer_10_20_time_series.png"

    plot_layer_time_series(
        simulation=simulation,
        observations=None,
        metrics=None,
        layer=RequestedLayer(10, 20),
        output_path=output_path,
        title_prefix="Best Trial",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_layer_scatter_creates_file(tmp_path):
    """Create an observed-versus-simulated plot for one soil layer."""

    _, _, matched = create_test_data()

    metrics = MetricsCalculator().calculate(matched)

    output_path = tmp_path / "layer_0_5_scatter.png"

    plot_layer_scatter(
        matched=matched,
        metrics=metrics,
        layer=RequestedLayer(0, 5),
        output_path=output_path,
        title_prefix="Best Trial",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
