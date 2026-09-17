import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from calibration import CalibrationRunner  # type: ignore
from parameters import SoilParameters  # type: ignore

MODEL_DIR = Path("model")
OBSERVATION_PATH = MODEL_DIR / "observation.csv"


def main() -> None:
    parameters = SoilParameters(
        ores=0.0875658533777091,
        osat=0.362417255147982,
        alfa=0.00603938365321607,
        npar=1.10657211418239,
        ksatfit=19.6334210990903,
        lexp=0.565929999579225,
        ksatexm=27.4823857626600,
        bdens=1428.27094863731,
    )

    runner = CalibrationRunner(
        model_dir=MODEL_DIR,
        observation_path=OBSERVATION_PATH,
        observation_layer=(0.0, 5.0),
    )

    result = runner.run(parameters)

    print("=" * 60)
    print("SWAP MATLAB Baseline")
    print("=" * 60)

    print("\nParameters")
    print("-" * 60)

    for name, value in parameters.__dict__.items():
        print(f"{name:10s}: {value:.10f}")

    print("\nData coverage")
    print("-" * 60)
    print(f"Field observations     : {result.n_observations}")
    print(f"Simulation dates       : {result.n_simulated}")
    print(f"Exact matched dates    : {result.n_matched}")
    print(f"Unmatched observations : {result.n_unmatched}")

    print("\nMetrics")
    print("-" * 60)
    print(f"RMSE : {result.metrics.rmse:.6f}")
    print(f"MAE  : {result.metrics.mae:.6f}")
    print(f"MBE  : {result.metrics.mbe:.6f}")
    print(f"R²   : {result.metrics.r2:.6f}")
    print(f"NSE  : {result.metrics.nse:.6f}")
    print(f"n    : {result.metrics.n}")

    print("\nMatched observations")
    print("-" * 60)
    print(
        result.matched.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


if __name__ == "__main__":
    main()
