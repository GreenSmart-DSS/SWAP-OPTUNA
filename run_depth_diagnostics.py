import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import OutputEvaluator  # type: ignore
from plotting import plot_layer_scatter, plot_layer_time_series  # type: ignore
from vap import RequestedColumn, RequestedLayer  # type: ignore

LAYERS = [
    RequestedLayer(0, 5),
    RequestedLayer(5, 10),
    RequestedLayer(10, 20),
    RequestedLayer(20, 30),
    RequestedLayer(5, 60),
]

COLUMNS = [
    RequestedColumn("wcontent", "weighted_mean"),
    RequestedColumn("phead", "weighted_mean"),
    RequestedColumn("hconduc", "weighted_mean"),
    RequestedColumn("temp", "weighted_mean"),
]


def main() -> None:
    """Evaluate and plot SWAP output for selected soil layers."""

    model_dir = PROJECT_DIR / "model"
    vap_path = model_dir / "results" / "result.vap"
    observation_path = model_dir / "observation.csv"

    output_dir = PROJECT_DIR / "depth_diagnostics"

    layers = LAYERS

    columns = COLUMNS

    observation_paths = {
        "0-5": observation_path,
    }

    evaluator = OutputEvaluator()

    results = evaluator.evaluate(
        vap_path=vap_path,
        layers=layers,
        observation_paths=observation_paths,
        columns=columns,
    )

    for result in results:
        layer_dir = output_dir / f"layer_{result.layer.top:g}_{result.layer.bottom:g}"

        layer_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.simulation.to_csv(layer_dir / "simulation.csv", index=False)

        plot_simulation = result.simulation[["date", "wcontent"]].rename(
            columns={"wcontent": "value"}
        )

        plot_layer_time_series(
            simulation=plot_simulation,
            observations=result.observations,
            metrics=result.metrics,
            layer=result.layer,
            output_path=layer_dir / "time_series.png",
            title_prefix=" ",
        )

        if result.metrics is not None and result.matched is not None:
            plot_layer_scatter(
                matched=result.matched,
                metrics=result.metrics,
                layer=result.layer,
                output_path=layer_dir / "scatter.png",
                title_prefix=" ",
            )

        print(f"\nLayer: {result.layer.name} cm")
        print(f"  Simulation records: {len(result.simulation)}")

        if result.observations is None:
            print("  Field observations: not available")
            print("  Metrics: not available")
        else:
            print(f"  Field observations: {len(result.observations)}")
            print(f"  Exact matches: {len(result.matched)}")  # type: ignore

            if result.metrics is None:
                print("  Metrics: not available")
            else:
                print(f"  RMSE: {result.metrics.rmse:.6f}")
                print(f"  MAE:  {result.metrics.mae:.6f}")
                print(f"  MBE:  {result.metrics.mbe:.6f}")
                print(f"  R2:   {result.metrics.r2:.6f}")
                print(f"  NSE:  {result.metrics.nse:.6f}")

        print(f"  Time series: {layer_dir / 'time_series.png'}")

        if result.metrics is not None:
            print(f"  Scatter:     {layer_dir / 'scatter.png'}")


if __name__ == "__main__":
    main()
