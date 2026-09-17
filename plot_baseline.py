import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from matching import ObservationMatcher  # type: ignore
from metrics import MetricsCalculator  # type: ignore
from observations import (  # type: ignore
    ObservationLayer,
    ObservationReader,
    ObservationVariable,
)
from plotting import plot_scatter, plot_time_series  # type: ignore
from vap import RequestedColumn, RequestedLayer, VAPReader  # type: ignore

MODEL_DIR = Path("model")
VAP_PATH = MODEL_DIR / "results" / "result.vap"
OBSERVATION_PATH = MODEL_DIR / "observation.csv"

FIGURE_DIR = Path("baseline_results")


def read_simulation() -> pd.DataFrame:
    """Read simulated 0-5 cm volumetric water content from VAP."""

    reader = VAPReader(VAP_PATH)

    simulation = reader.extract(
        layers=[
            RequestedLayer(
                top=0.0,
                bottom=5.0,
            )
        ],
        columns=[
            RequestedColumn(
                name="wcontent",
                aggregation="weighted_mean",
            )
        ],
    )

    simulation = simulation[simulation["layer"] == "0-5"].copy()

    return (
        simulation[["date", "wcontent"]]
        .rename(columns={"wcontent": "value"})
        .sort_values("date")
        .reset_index(drop=True)
    )


def read_observations() -> pd.DataFrame:
    """Read field observations."""

    reader = ObservationReader(
        path=OBSERVATION_PATH,
        variable=ObservationVariable(
            name="wcontent",
            unit="cm3/cm3",
        ),
        layer=ObservationLayer(
            top=0.0,
            bottom=5.0,
        ),
    )

    return reader.read()


def main() -> None:
    if not VAP_PATH.exists():
        raise FileNotFoundError(f"SWAP VAP output was not found: {VAP_PATH}")

    if not OBSERVATION_PATH.exists():
        raise FileNotFoundError(f"Observation file was not found: {OBSERVATION_PATH}")

    simulation = read_simulation()
    observations = read_observations()

    matcher = ObservationMatcher()
    matched = matcher.match(
        observations,
        simulation,
    )

    metrics = MetricsCalculator().calculate(matched)

    FIGURE_DIR.mkdir(exist_ok=True)

    plot_time_series(
        simulation=simulation,
        observations=observations,
        metrics=metrics,
        output_path=FIGURE_DIR / "baseline_time_series.png",
    )

    plot_scatter(
        matched=matched,
        metrics=metrics,
        output_path=FIGURE_DIR / "baseline_scatter.png",
    )

    print("=" * 60)
    print("Baseline figures generated")
    print("=" * 60)
    print(f"Simulation dates : {len(simulation)}")
    print(f"Observations     : {len(observations)}")
    print(f"Exact matches    : {len(matched)}")
    print()
    print(f"RMSE : {metrics.rmse:.6f}")
    print(f"MAE  : {metrics.mae:.6f}")
    print(f"MBE  : {metrics.mbe:.6f}")
    print(f"R²   : {metrics.r2:.6f}")
    print(f"NSE  : {metrics.nse:.6f}")
    print(f"n    : {metrics.n}")
    print()
    print(f"Time series : {FIGURE_DIR / 'baseline_time_series.png'}")
    print(f"Scatter     : {FIGURE_DIR / 'baseline_scatter.png'}")


if __name__ == "__main__":
    main()
